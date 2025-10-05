import os
import time
from datetime import datetime

import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Numeric
from sqlalchemy.orm import declarative_base, sessionmaker

from rag_core import rag, calculate_llm_cost, compute_relevancy  # must return dict

st.set_page_config(page_title="RAG Q&A", page_icon="🧠", layout="centered")

# ---------------------
# Postgres configuration
# ---------------------
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB = os.getenv("POSTGRES_DB", "rag_metrics")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

DB_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

# SQLAlchemy setup
Base = declarative_base()

def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)

_engine = None
_SessionLocal = None

def get_session():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = get_engine()
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_engine)
    return _SessionLocal()

# ORM model
class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    feedback = Column(String(10), nullable=True)  # 'up' | 'down' | None
    latency_ms = Column(Float, nullable=True)
    tokens_in = Column(Integer, nullable=True)
    tokens_out = Column(Integer, nullable=True)
    cost = Column(Numeric(10, 6), nullable=True)       # USD cost
    relevancy = Column(String, nullable=True)          # e.g. "RELEVANT"
    relevancy_explanation = Column(Text, nullable=True) # explanation

# Create table if it doesn't exist (do this once at app start)
try:
    eng = get_engine()
    Base.metadata.create_all(eng)
except Exception as e:
    st.sidebar.warning(f"DB not ready yet: {e}")

# ---------------------
# State
# ---------------------
if "last_row_id" not in st.session_state:
    st.session_state.last_row_id = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = None

# ---------------------
# UI
# ---------------------
st.title("🧠 RAG Q&A")
query = st.text_input("Enter your question:")

ask_clicked = st.button("Ask")
if ask_clicked:
    if query.strip():
        t0 = time.perf_counter()
        with st.spinner("Thinking with RAG…"):
            try:
                rag_result = rag(query)  # dict: answer, tokens_in, tokens_out
                answer_text = rag_result.get("answer", "")
                tokens_in = int(rag_result.get("tokens_in", 0) or 0)
                tokens_out = int(rag_result.get("tokens_out", 0) or 0)
                cost = calculate_llm_cost(tokens_in, tokens_out)

                # Compute relevancy (expected dict with "Relevance" & "Explanation")
                rel = compute_relevancy(query, answer_text)
                relevancy = None
                relevancy_expl = None
                if isinstance(rel, dict):
                    relevancy = str(rel.get("Relevance"))
                    relevancy_expl = str(rel.get("Explanation"))
            except Exception as e:
                answer_text = f"❌ Error calling rag(): {e}"
                tokens_in, tokens_out, cost = 0, 0, None
                relevancy, relevancy_expl = None, None
        latency = (time.perf_counter() - t0) * 1000.0

        # Persist Q&A
        try:
            session = get_session()
            row = Interaction(
                question=query,
                answer=str(answer_text),
                latency_ms=latency,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
                relevancy=relevancy,
                relevancy_explanation=relevancy_expl,
            )
            session.add(row)
            session.commit()
            st.session_state.last_row_id = row.id
            st.session_state.last_answer = str(answer_text)
            st.session_state.last_question = query
            st.session_state.last_feedback = None
        except Exception as db_err:
            st.error(f"Failed to save interaction: {db_err}")
        finally:
            try:
                session.close()
            except Exception:
                pass
    else:
        st.warning("Please enter a question before asking.")

# Always show the latest answer + feedback controls after any rerun
if st.session_state.last_answer is not None:
    st.write("### Answer")
    st.write(st.session_state.last_answer)

    if st.session_state.last_row_id:
        try:
            session = get_session()
            obj = session.get(Interaction, st.session_state.last_row_id)
            if obj:
                st.caption(
                    f"Tokens in: {obj.tokens_in or 0} • "
                    f"Tokens out: {obj.tokens_out or 0} • "
                    f"Cost: ${format(obj.cost or 0, '.6f')} • "
                    f"Relevancy: {obj.relevancy or '-'} • "
                    f"Explanation: {obj.relevancy_explanation or '-'} • "
                    f"Latency: {obj.latency_ms:.1f} ms"
                )
        except Exception:
            pass
        finally:
            try:
                session.close()
            except Exception:
                pass

    col1, col2 = st.columns(2)

    up_disabled = st.session_state.last_feedback == "up"
    down_disabled = st.session_state.last_feedback == "down"

    with col1:
        if st.button("👍 Helpful", key="fb_up", disabled=up_disabled):
            if st.session_state.last_row_id:
                try:
                    session = get_session()
                    obj = session.get(Interaction, st.session_state.last_row_id)
                    if obj:
                        obj.feedback = "up"
                        session.commit()
                        st.session_state.last_feedback = "up"
                        st.success("Thanks for your feedback! 👍")
                except Exception as e:
                    st.error(f"Failed to save feedback: {e}")
                finally:
                    try:
                        session.close()
                    except Exception:
                        pass
    with col2:
        if st.button("👎 Not Helpful", key="fb_down", disabled=down_disabled):
            if st.session_state.last_row_id:
                try:
                    session = get_session()
                    obj = session.get(Interaction, st.session_state.last_row_id)
                    if obj:
                        obj.feedback = "down"
                        session.commit()
                        st.session_state.last_feedback = "down"
                        st.info("Thanks for letting us know. 👎")
                except Exception as e:
                    st.error(f"Failed to save feedback: {e}")
                finally:
                    try:
                        session.close()
                    except Exception:
                        pass

# Helpful footer
st.caption(
    f"DB: postgresql://{PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DB} — "
    f"Last row id: {st.session_state.last_row_id} — "
    f"Feedback: {st.session_state.last_feedback}"
)