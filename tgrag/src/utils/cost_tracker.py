"""Cost tracking for LLM and embedding operations."""

from __future__ import annotations

import functools
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional, Dict, List

logger = logging.getLogger("temporal-graphrag.cost")


class CostTracker:
    """
    Tracks token usage, call counts, cache hits/misses, and provides
    async-compatible decorators to wrap LLM and embedding calls.
    Also handles configuration tracking and comprehensive result generation.
    """

    def __init__(self) -> None:
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_embedding_tokens: int = 0
        self.total_tokens: int = 0
        self.call_count: int = 0
        self.start_time: float = time.time()
        self.llm_calls: int = 0
        self.embedding_calls: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        
        # Additional tracking for comprehensive results
        self.configuration: Optional[Dict[str, Any]] = None
        self.corpus_info: Optional[Dict[str, Any]] = None
        self.build_metadata: Optional[Dict[str, Any]] = None
        self.querying_metadata: Optional[Dict[str, Any]] = None
        
        # Querying-specific tracking
        self.total_queries_processed: int = 0
        self.successful_queries: int = 0
        self.failed_queries: int = 0
        self.enhanced_queries: int = 0
        self.standard_queries: int = 0
        
        # Incremental update tracking
        self.is_incremental: bool = False
        self.existing_docs_count: int = 0
        self.new_docs_count: int = 0
        self.preserve_communities: bool = False
        self.reused_embeddings: int = 0
        self.reused_community_summaries: int = 0
        self.saved_tokens_from_reuse: int = 0
        self.saved_cost_from_reuse: float = 0.0

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        self.cache_misses += 1
    
    def record_reused_embeddings(self, count: int) -> None:
        """Record reused embeddings from incremental updates"""
        self.reused_embeddings += count
    
    def record_reused_community_summaries(self, count: int) -> None:
        """Record reused community summaries from incremental updates"""
        self.reused_community_summaries += count
    
    def record_saved_tokens(self, tokens: int, estimated_cost: float = 0.0) -> None:
        """Record tokens and cost saved from reuse"""
        self.saved_tokens_from_reuse += tokens
        self.saved_cost_from_reuse += estimated_cost
    
    def record_query_processed(self, success: bool = True, enhanced: bool = False) -> None:
        """Record a processed query"""
        self.total_queries_processed += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        
        if enhanced:
            self.enhanced_queries += 1
        else:
            self.standard_queries += 1
    
    def get_querying_stats(self) -> Dict[str, Any]:
        """Get querying-specific statistics"""
        if self.total_queries_processed == 0:
            return {}
        
        success_rate = (self.successful_queries / self.total_queries_processed) * 100.0
        enhanced_rate = (self.enhanced_queries / self.total_queries_processed) * 100.0
        
        return {
            "total_queries_processed": self.total_queries_processed,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "enhanced_queries": self.enhanced_queries,
            "standard_queries": self.standard_queries,
            "success_rate_percent": round(success_rate, 2),
            "enhanced_usage_rate_percent": round(enhanced_rate, 2)
        }
    
    def set_configuration(self, config: Dict[str, Any], model: str, effective_working_dir: str) -> None:
        """Set the build configuration for tracking"""
        # Get provider, with backward compatibility for use_google
        provider = config.get('provider')
        if provider is None:
            # Infer from model or use legacy use_google flag
            from ..config.config_loader import infer_provider_from_model
            use_google = config.get('use_google')
            provider = infer_provider_from_model(model, use_google)
        
        self.configuration = {
            "corpus_path": config.get('corpus_path', './ECT_data/subset'),
            "working_dir": effective_working_dir,
            "model": model,
            "baseline": config.get('baseline', 'temporalrag'),
            "provider": provider,
            "chunk_size": config.get('chunk_size', 1200),
            "chunk_overlap": config.get('chunk_overlap', 100),
            "enable_seasonal_matching": config.get('enable_seasonal_matching', False),
            "embedding_settings": {
                "embedding_dim": 1536,
                "max_token_size": 8192,
                "embedding_func_max_async": 4
            }
        }
    
    def set_corpus_info(self, text_files: List[str]) -> None:
        """Set corpus information for tracking"""
        self.corpus_info = {
            "total_files": len(text_files),
            "file_names": text_files
        }
    
    def set_build_metadata(self, effective_working_dir: str) -> None:
        """Set build metadata for tracking"""
        self.build_metadata = {
            "build_timestamp": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "graph_storage_path": os.path.join('./graph_storage', effective_working_dir),
            "vector_db_storage_class": "NumericVectorDBStorage"
        }

    def set_incremental_build_info(self, is_incremental: bool, existing_docs_count: int, 
                                  new_docs_count: int, preserve_communities: bool) -> None:
        """Set incremental build information for tracking"""
        self.is_incremental = is_incremental
        self.existing_docs_count = existing_docs_count
        self.new_docs_count = new_docs_count
        self.preserve_communities = preserve_communities
        
        if is_incremental:
            logger.info(f"Incremental build mode enabled:")
            logger.info(f"  - Existing documents: {existing_docs_count}")
            logger.info(f"  - New documents: {new_docs_count}")
            logger.info(f"  - Community preservation: {'ENABLED' if preserve_communities else 'DISABLED'}")
        else:
            logger.info("Full rebuild mode - processing all documents")

    def set_incremental_mode(self, is_incremental: bool, existing_docs_count: int) -> None:
        """Set incremental mode for backward compatibility"""
        self.is_incremental = is_incremental
        self.existing_docs_count = existing_docs_count

    def track_generated_tokens(self, func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                self.llm_calls += 1
                response = await func(*args, **kwargs)

                # Expecting either raw string or tuple of (result, usage_dict)
                if isinstance(response, tuple) and len(response) >= 2 and isinstance(response[1], dict):
                    usage = response[1]
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
                    completion_tokens = int(usage.get("completion_tokens", 0))
                    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
                    
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_tokens += total_tokens
                    self.call_count += 1
                    
                    print(f"🔍 Cost tracker: {func.__name__} - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                else:
                    # If no usage data, still count the call but with estimated tokens
                    print(f"⚠️  Warning: LLM function {func.__name__} returned unexpected format: {type(response)}")
                    self.call_count += 1
                    
                return response
            except Exception as e:
                print(f"❌ Error in cost tracker for {func.__name__}: {str(e)}")
                # Still count the call even if it failed
                self.llm_calls += 1
                self.call_count += 1
                raise  # Re-raise the exception

        return wrapper

    def track_embedding_tokens(self, func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                self.embedding_calls += 1
                response = await func(*args, **kwargs)
                # Expecting (embeddings, usage_dict)
                if isinstance(response, tuple) and len(response) >= 2 and isinstance(response[1], dict):
                    usage = response[1]
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
                    total_tokens = int(usage.get("total_tokens", prompt_tokens))
                    
                    self.total_embedding_tokens += prompt_tokens
                    self.total_tokens += total_tokens
                    self.call_count += 1
                    
                    print(f"🔍 Cost tracker: {func.__name__} - Embedding tokens: {prompt_tokens}, Total: {total_tokens}")
                else:
                    # If no usage data, still count the call but with estimated tokens
                    print(f"⚠️  Warning: Embedding function {func.__name__} returned unexpected format: {type(response)}")
                    self.call_count += 1
                    
                return response
            except Exception as e:
                print(f"❌ Error in cost tracker for {func.__name__}: {str(e)}")
                # Still count the call even if it failed
                self.embedding_calls += 1
                self.call_count += 1
                raise  # Re-raise the exception

        return wrapper

    def get_totals(self) -> dict:
        end_time = time.time()
        execution_time = end_time - self.start_time

        execution_time_formatted = str(timedelta(seconds=int(execution_time)))
        tokens_per_second = self.total_tokens / execution_time if execution_time > 0 else 0.0
        calls_per_second = self.call_count / execution_time if execution_time > 0 else 0.0

        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_cache_ops * 100.0) if total_cache_ops > 0 else 0.0

        # Ensure total_tokens is calculated correctly
        calculated_total_tokens = self.total_prompt_tokens + self.total_completion_tokens + self.total_embedding_tokens
        
        # Add incremental update statistics
        incremental_stats = {}
        if self.is_incremental:
            incremental_stats = {
                "incremental_mode": True,
                "existing_docs_count": self.existing_docs_count,
                "new_docs_count": self.new_docs_count,
                "preserve_communities": self.preserve_communities,
                "reused_embeddings": self.reused_embeddings,
                "reused_community_summaries": self.reused_community_summaries,
                "saved_tokens_from_reuse": self.saved_tokens_from_reuse,
                "saved_cost_from_reuse": round(self.saved_cost_from_reuse, 4),
                "efficiency_gain_percent": round((self.saved_tokens_from_reuse / (calculated_total_tokens + self.saved_tokens_from_reuse)) * 100, 2) if (calculated_total_tokens + self.saved_tokens_from_reuse) > 0 else 0.0
            }
        else:
            incremental_stats = {
                "incremental_mode": False,
                "existing_docs_count": 0,
                "new_docs_count": self.new_docs_count,
                "preserve_communities": False,
                "reused_embeddings": 0,
                "reused_community_summaries": 0,
                "saved_tokens_from_reuse": 0,
                "saved_cost_from_reuse": 0.0,
                "efficiency_gain_percent": 0.0
            }

        # Rough cost estimates (adjust as needed)
        estimated_cost = (
            (self.total_prompt_tokens / 1000.0) * 0.0015
            + (self.total_completion_tokens / 1000.0) * 0.002
            + (self.total_embedding_tokens / 1000.0) * 0.00002
        )

        return {
            "total_calls": self.call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_embedding_tokens": self.total_embedding_tokens,
            "grand_total_tokens": self.total_tokens,
            "execution_start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "execution_end_time": datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "total_execution_time_seconds": round(execution_time, 2),
            "total_execution_time_formatted": execution_time_formatted,
            "tokens_per_second": round(tokens_per_second, 2),
            "calls_per_second": round(calls_per_second, 2),
            "average_tokens_per_call": round(self.total_tokens / self.call_count, 2) if self.call_count > 0 else 0.0,
            "llm_calls": self.llm_calls,
            "embedding_calls": self.embedding_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "estimated_cost_usd": round(estimated_cost, 4),
            "prompt_to_completion_ratio": round(
                self.total_prompt_tokens / self.total_completion_tokens, 2
            ) if self.total_completion_tokens > 0 else 0.0,
            **incremental_stats
        }
    
    def get_comprehensive_results(self) -> Dict[str, Any]:
        """Get comprehensive build results including configuration and metadata"""
        execution_summary = self.get_totals()
        
        # Add querying stats if available
        querying_stats = {}
        if self.querying_metadata:
            querying_stats = self.get_querying_stats()
        
        return {
            "execution_summary": execution_summary,
            "configuration": self.configuration or {},
            "corpus_info": self.corpus_info or {},
            "build_metadata": self.build_metadata or {},
            "querying_metadata": self.querying_metadata or {},
            "querying_stats": querying_stats
        }
    
    def save_results(self, output_dir: str, save_legacy: bool = True, operation_type: str = "build") -> tuple[str, str]:
        """
        Save comprehensive results to files with descriptive naming
        
        Args:
            output_dir: Directory to save results to
            save_legacy: Whether to also save legacy format file
            operation_type: Type of operation ("build" or "query")
            
        Returns:
            Tuple of (comprehensive_results_path, legacy_results_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get execution summary for naming
        execution_summary = self.get_totals()
        
        # Generate descriptive filename based on operation type
        if self.configuration:
            model_safe = self.configuration.get('model', 'unknown').replace('/', '_').replace(':', '_')
            corpus_path = self.configuration.get('corpus_path', './ECT_data/subset')
            corpus_name = os.path.basename(os.path.normpath(corpus_path))
        else:
            model_safe = 'unknown'
            corpus_name = 'unknown'
        
        timestamp = execution_summary['execution_start_time'].replace(':', '').replace('-', '').replace(' ', '_')
        
        if operation_type == "query":
            result_filename = f"querying_results_{corpus_name}_{model_safe}_{timestamp}.json"
        else:
            result_filename = f"build_results_{corpus_name}_{model_safe}_{timestamp}.json"
        
        # Save comprehensive results
        comprehensive_path = os.path.join(output_dir, result_filename)
        comprehensive_results = self.get_comprehensive_results()
        
        with open(comprehensive_path, "w", encoding="utf-8") as f:
            json.dump(comprehensive_results, f, indent=4, ensure_ascii=False)
        
        # Save legacy format if requested
        legacy_path = ""
        if save_legacy:
            if operation_type == "query":
                legacy_path = os.path.join(output_dir, "querying_token_consumption.json")
            else:
                legacy_path = os.path.join(output_dir, "build_graph_token_consumption.json")
            
            with open(legacy_path, "w", encoding="utf-8") as f:
                json.dump([execution_summary], f, indent=4, ensure_ascii=False)
        
        return comprehensive_path, legacy_path
    
    def save_querying_results(self, output_dir: str, save_legacy: bool = True) -> tuple[str, str]:
        """
        Save querying-specific results to files with appropriate naming
        
        Args:
            output_dir: Directory to save results to
            save_legacy: Whether to also save legacy format file
            
        Returns:
            Tuple of (comprehensive_results_path, legacy_results_path)
        """
        return self.save_results(output_dir, save_legacy, operation_type="query")
    
    def print_summary(self) -> None:
        """Print a formatted summary of the execution results"""
        totals = self.get_totals()
        
        # Determine operation type for header
        operation_type = "QUERYING" if self.querying_metadata else "BUILDING"
        
        print("\n" + "="*60)
        print(f"📊 TEMPORAL GRAPH {operation_type} SUMMARY")
        print("="*60)
        
        # Time Information
        print(f"\n⏱️  EXECUTION TIME:")
        print(f"   Start: {totals['execution_start_time']}")
        print(f"   End:   {totals['execution_end_time']}")
        print(f"   Duration: {totals['total_execution_time_formatted']} ({totals['total_execution_time_seconds']} seconds)")
        
        # Token Usage
        print(f"\n🔢 TOKEN USAGE:")
        print(f"   Total Calls: {totals['total_calls']:,}")
        print(f"   Prompt Tokens: {totals['total_prompt_tokens']:,}")
        print(f"   Completion Tokens: {totals['total_completion_tokens']:,}")
        print(f"   Embedding Tokens: {totals['total_embedding_tokens']:,}")
        print(f"   Grand Total: {totals['grand_total_tokens']:,}")
        
        # Performance Metrics
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Tokens/Second: {totals['tokens_per_second']:,}")
        print(f"   Calls/Second: {totals['calls_per_second']:.2f}")
        print(f"   Avg Tokens/Call: {totals['average_tokens_per_call']:.1f}")
        
        # Call Breakdown
        print(f"\n📞 CALL BREAKDOWN:")
        print(f"   LLM Calls: {totals['llm_calls']:,}")
        print(f"   Embedding Calls: {totals['embedding_calls']:,}")
        
        # Cache Performance
        if totals['cache_hits'] > 0 or totals['cache_misses'] > 0:
            print(f"\n💾 CACHE PERFORMANCE:")
            print(f"   Cache Hits: {totals['cache_hits']:,}")
            print(f"   Cache Misses: {totals['cache_misses']:,}")
            print(f"   Hit Rate: {totals['cache_hit_rate_percent']:.1f}%")
        
        # Querying-specific information
        if self.querying_metadata:
            print(f"\n🔍 QUERYING CONFIGURATION:")
            print(f"   Evaluation Mode: {self.querying_metadata.get('evaluation_mode', 'unknown')}")
            print(f"   Baseline: {self.querying_metadata.get('baseline', 'unknown')}")
            print(f"   Enhanced Retrieval: {self.querying_metadata.get('use_enhanced_retrieval', False)}")
            print(f"   Top-K: {self.querying_metadata.get('top_k', 'unknown')}")
            print(f"   Subgraph Enabled: {self.querying_metadata.get('enable_subgraph', 'unknown')}")
            print(f"   Mixed Relations: {self.querying_metadata.get('enable_mixed_relationship', 'unknown')}")
            
            # Show querying statistics if available
            querying_stats = self.get_querying_stats()
            if querying_stats:
                print(f"\n📊 QUERYING STATISTICS:")
                print(f"   Total Queries: {querying_stats.get('total_queries_processed', 0):,}")
                print(f"   Successful: {querying_stats.get('successful_queries', 0):,}")
                print(f"   Failed: {querying_stats.get('failed_queries', 0):,}")
                print(f"   Enhanced: {querying_stats.get('enhanced_queries', 0):,}")
                print(f"   Standard: {querying_stats.get('standard_queries', 0):,}")
                print(f"   Success Rate: {querying_stats.get('success_rate_percent', 0):.1f}%")
                print(f"   Enhanced Usage: {querying_stats.get('enhanced_usage_rate_percent', 0):.1f}%")
        
        # Cost Estimation
        print(f"\n💰 ESTIMATED COST:")
        print(f"   Total Cost: ${totals['estimated_cost_usd']:.4f}")
        
        # Efficiency Metrics
        print(f"\n📈 EFFICIENCY METRICS:")
        print(f"   Prompt/Completion Ratio: {totals['prompt_to_completion_ratio']:.2f}")
        
        print("\n" + "="*60)

