
"""
Rate limiting service for API protection
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
import redis
from ..config import settings

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting service using Redis"""
    
    def __init__(self):
        self.redis_client = None
        self.local_cache = {}  # Fallback for when Redis is unavailable
        
    async def _get_redis_client(self):
        """Get Redis client (lazy initialization)"""
        if self.redis_client is None:
            try:
                self.redis_client = redis.from_url(settings.REDIS_URL)
                await self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis unavailable, using local cache: {str(e)}")
                self.redis_client = None
        return self.redis_client
    
    async def check_rate_limit(self, client_ip: str, user_id: str) -> bool:
        """Check if request is within rate limits"""
        try:
            redis_client = await self._get_redis_client()
            
            if redis_client:
                return await self._check_redis_rate_limit(redis_client, client_ip, user_id)
            else:
                return self._check_local_rate_limit(client_ip, user_id)
                
        except Exception as e:
            logger.error(f"Rate limit check failed: {str(e)}")
            return True  # Allow request if rate limiting fails
    
    async def check_batch_rate_limit(self, client_ip: str, user_id: str, batch_size: int) -> bool:
        """Check batch rate limits"""
        try:
            # Check if batch size is within limits
            if batch_size > settings.MAX_BATCH_SIZE:
                return False
            
            redis_client = await self._get_redis_client()
            
            if redis_client:
                return await self._check_redis_batch_rate_limit(redis_client, client_ip, user_id, batch_size)
            else:
                return self._check_local_batch_rate_limit(client_ip, user_id, batch_size)
                
        except Exception as e:
            logger.error(f"Batch rate limit check failed: {str(e)}")
            return True
    
    async def _check_redis_rate_limit(self, redis_client, client_ip: str, user_id: str) -> bool:
        """Check rate limits using Redis"""
        try:
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(minutes=1)
            
            # Create keys for IP and user
            ip_key = f"rate_limit:ip:{client_ip}"
            user_key = f"rate_limit:user:{user_id}"
            
            # Use Redis pipeline for atomic operations
            pipe = redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(ip_key, 0, window_start.timestamp())
            pipe.zremrangebyscore(user_key, 0, window_start.timestamp())
            
            # Count current requests
            pipe.zcard(ip_key)
            pipe.zcard(user_key)
            
            # Add current request
            pipe.zadd(ip_key, {str(current_time.timestamp()): current_time.timestamp()})
            pipe.zadd(user_key, {str(current_time.timestamp()): current_time.timestamp()})
            
            # Set expiration
            pipe.expire(ip_key, 3600)  # 1 hour
            pipe.expire(user_key, 3600)  # 1 hour
            
            results = await pipe.execute()
            
            ip_count = results[2]
            user_count = results[3]
            
            # Check limits
            if ip_count >= settings.RATE_LIMIT_PER_MINUTE:
                logger.warning(f"IP rate limit exceeded: {client_ip}")
                return False
            
            if user_count >= settings.RATE_LIMIT_PER_MINUTE:
                logger.warning(f"User rate limit exceeded: {user_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {str(e)}")
            return True
    
    async def _check_redis_batch_rate_limit(self, redis_client, client_ip: str, user_id: str, batch_size: int) -> bool:
        """Check batch rate limits using Redis"""
        try:
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(minutes=1)
            
            # Create keys for batch limits
            ip_batch_key = f"batch_rate_limit:ip:{client_ip}"
            user_batch_key = f"batch_rate_limit:user:{user_id}"
            
            pipe = redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(ip_batch_key, 0, window_start.timestamp())
            pipe.zremrangebyscore(user_batch_key, 0, window_start.timestamp())
            
            # Count current batch requests
            pipe.zcard(ip_batch_key)
            pipe.zcard(user_batch_key)
            
            results = await pipe.execute()
            
            ip_batch_count = results[2]
            user_batch_count = results[3]
            
            # Check batch limits (stricter than regular limits)
            if ip_batch_count >= settings.BATCH_RATE_LIMIT_PER_MINUTE:
                logger.warning(f"IP batch rate limit exceeded: {client_ip}")
                return False
            
            if user_batch_count >= settings.BATCH_RATE_LIMIT_PER_MINUTE:
                logger.warning(f"User batch rate limit exceeded: {user_id}")
                return False
            
            # Add current batch request
            pipe.zadd(ip_batch_key, {str(current_time.timestamp()): current_time.timestamp()})
            pipe.zadd(user_batch_key, {str(current_time.timestamp()): current_time.timestamp()})
            pipe.expire(ip_batch_key, 3600)
            pipe.expire(user_batch_key, 3600)
            
            await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Redis batch rate limit check failed: {str(e)}")
            return True
    
    def _check_local_rate_limit(self, client_ip: str, user_id: str) -> bool:
        """Check rate limits using local cache (fallback)"""
        try:
            current_time = datetime.utcnow()
            
            # Clean old entries
            self._clean_local_cache()
            
            # Check IP rate limit
            ip_key = f"ip:{client_ip}"
            if ip_key not in self.local_cache:
                self.local_cache[ip_key] = []
            
            # Remove old entries for this IP
            self.local_cache[ip_key] = [
                timestamp for timestamp in self.local_cache[ip_key]
                if current_time - timestamp < timedelta(minutes=1)
            ]
            
            if len(self.local_cache[ip_key]) >= settings.RATE_LIMIT_PER_MINUTE:
                return False
            
            # Check user rate limit
            user_key = f"user:{user_id}"
            if user_key not in self.local_cache:
                self.local_cache[user_key] = []
            
            # Remove old entries for this user
            self.local_cache[user_key] = [
                timestamp for timestamp in self.local_cache[user_key]
                if current_time - timestamp < timedelta(minutes=1)
            ]
            
            if len(self.local_cache[user_key]) >= settings.RATE_LIMIT_PER_MINUTE:
                return False
            
            # Add current request
            self.local_cache[ip_key].append(current_time)
            self.local_cache[user_key].append(current_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Local rate limit check failed: {str(e)}")
            return True
    
    def _check_local_batch_rate_limit(self, client_ip: str, user_id: str, batch_size: int) -> bool:
        """Check batch rate limits using local cache"""
        try:
            current_time = datetime.utcnow()
            
            # Check batch-specific limits
            ip_batch_key = f"batch_ip:{client_ip}"
            user_batch_key = f"batch_user:{user_id}"
            
            if ip_batch_key not in self.local_cache:
                self.local_cache[ip_batch_key] = []
            if user_batch_key not in self.local_cache:
                self.local_cache[user_batch_key] = []
            
            # Clean old entries
            self.local_cache[ip_batch_key] = [
                timestamp for timestamp in self.local_cache[ip_batch_key]
                if current_time - timestamp < timedelta(minutes=1)
            ]
            self.local_cache[user_batch_key] = [
                timestamp for timestamp in self.local_cache[user_batch_key]
                if current_time - timestamp < timedelta(minutes=1)
            ]
            
            # Check limits
            if len(self.local_cache[ip_batch_key]) >= settings.BATCH_RATE_LIMIT_PER_MINUTE:
                return False
            if len(self.local_cache[user_batch_key]) >= settings.BATCH_RATE_LIMIT_PER_MINUTE:
                return False
            
            # Add current batch request
            self.local_cache[ip_batch_key].append(current_time)
            self.local_cache[user_batch_key].append(current_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Local batch rate limit check failed: {str(e)}")
            return True
    
    def _clean_local_cache(self):
        """Clean old entries from local cache"""
        try:
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(hours=1)
            
            keys_to_remove = []
            for key, timestamps in self.local_cache.items():
                # Remove old timestamps
                self.local_cache[key] = [
                    timestamp for timestamp in timestamps
                    if timestamp > cutoff_time
                ]
                
                # Remove empty keys
                if not self.local_cache[key]:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.local_cache[key]
                
        except Exception as e:
            logger.error(f"Local cache cleanup failed: {str(e)}")
    
    async def get_rate_limit_status(self, client_ip: str, user_id: str) -> Dict[str, int]:
        """Get current rate limit status"""
        try:
            redis_client = await self._get_redis_client()
            
            if redis_client:
                return await self._get_redis_rate_limit_status(redis_client, client_ip, user_id)
            else:
                return self._get_local_rate_limit_status(client_ip, user_id)
                
        except Exception as e:
            logger.error(f"Failed to get rate limit status: {str(e)}")
            return {"ip_requests": 0, "user_requests": 0, "limit": settings.RATE_LIMIT_PER_MINUTE}
    
    async def _get_redis_rate_limit_status(self, redis_client, client_ip: str, user_id: str) -> Dict[str, int]:
        """Get rate limit status from Redis"""
        try:
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(minutes=1)
            
            ip_key = f"rate_limit:ip:{client_ip}"
            user_key = f"rate_limit:user:{user_id}"
            
            pipe = redis_client.pipeline()
            pipe.zremrangebyscore(ip_key, 0, window_start.timestamp())
            pipe.zremrangebyscore(user_key, 0, window_start.timestamp())
            pipe.zcard(ip_key)
            pipe.zcard(user_key)
            
            results = await pipe.execute()
            
            return {
                "ip_requests": results[2],
                "user_requests": results[3],
                "limit": settings.RATE_LIMIT_PER_MINUTE,
                "window_minutes": 1
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis rate limit status: {str(e)}")
            return {"ip_requests": 0, "user_requests": 0, "limit": settings.RATE_LIMIT_PER_MINUTE}
    
    def _get_local_rate_limit_status(self, client_ip: str, user_id: str) -> Dict[str, int]:
        """Get rate limit status from local cache"""
        try:
            current_time = datetime.utcnow()
            
            ip_key = f"ip:{client_ip}"
            user_key = f"user:{user_id}"
            
            ip_requests = 0
            user_requests = 0
            
            if ip_key in self.local_cache:
                ip_requests = len([
                    timestamp for timestamp in self.local_cache[ip_key]
                    if current_time - timestamp < timedelta(minutes=1)
                ])
            
            if user_key in self.local_cache:
                user_requests = len([
                    timestamp for timestamp in self.local_cache[user_key]
                    if current_time - timestamp < timedelta(minutes=1)
                ])
            
            return {
                "ip_requests": ip_requests,
                "user_requests": user_requests,
                "limit": settings.RATE_LIMIT_PER_MINUTE,
                "window_minutes": 1
            }
            
        except Exception as e:
            logger.error(f"Failed to get local rate limit status: {str(e)}")
            return {"ip_requests": 0, "user_requests": 0, "limit": settings.RATE_LIMIT_PER_MINUTE}
