# 🧠 OmniBrain — Database & Vector Store Architecture Documentation

Welcome to the **OmniBrain Database & Vector Store Subsystem** documentation. This document details the relational database design, Qdrant vector database topology, system-wide Pydantic data contracts (DTOs), and CRUD helper interfaces for all team members.

---

## 🏗️ 1. System Architecture Overview

OmniBrain utilizes a dual-engine storage architecture:
1. **Relational Database (SQLAlchemy 2.0 Async + PostgreSQL / SQLite)**: Stores user credentials, document metadata, multi-turn chat sessions, message histories with citation links, and immutable audit logs.
2. **Vector Database (Qdrant)**: Stores dense embeddings for fast cosine similarity search across text chunks and multi-modal image representations.

```mermaid
erDiagram
    USERS ||--o{ DOCUMENTS : "owns"
    USERS ||--o{ CHAT_SESSIONS : "creates"
    USERS ||--o{ AUDIT_LOGS : "triggers"
    CHAT_SESSIONS ||--o{ CHAT_MESSAGES : "contains"
    DOCUMENTS ||--o{ CHAT_MESSAGES : "cited in"

    USERS {
        uuid id PK
        string email UK
        string username UK
        string full_name
        string hashed_password
        string role
        boolean is_active
        datetime created_at
        datetime updated_at
    }

    DOCUMENTS {
        uuid id PK
        uuid user_id FK
        string title
        string file_url
        string file_type
        int file_size_bytes
        int page_count
        string status
        json tags
        json meta_info
        int chunk_count
        datetime created_at
        datetime updated_at
    }

    CHAT_SESSIONS {
        uuid id PK
        uuid user_id FK
        string title
        text system_prompt
        boolean is_archived
        json meta_info
        datetime created_at
        datetime updated_at
    }

    CHAT_MESSAGES {
        uuid id PK
        uuid session_id FK
        string role
        text content
        string image_url
        int tokens_used
        json citations
        json meta_info
        datetime created_at
    }

    AUDIT_LOGS {
        uuid id PK
        uuid user_id FK
        string action
        string resource_type
        string resource_id
        json details
        string ip_address
        datetime timestamp
    }