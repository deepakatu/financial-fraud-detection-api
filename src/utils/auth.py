
"""
Authentication utilities
"""

import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException
from ..config import settings

logger = logging.getLogger(__name__)

def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    try:
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        return encoded_jwt
        
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        raise

def verify_token(token: str) -> str:
    """Verify JWT token and return user ID"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Check token type
        token_type = payload.get("type")
        if token_type != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        return user_id
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Error verifying token: {str(e)}")
        raise HTTPException(status_code=401, detail="Token verification failed")

def create_demo_token() -> str:
    """Create demo token for testing"""
    return create_access_token("demo_user")
