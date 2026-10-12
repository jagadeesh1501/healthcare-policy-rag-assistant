import time, streamlit as st
from v4_reranking import ask
st.set_page_config(page_title='Health Policy RAG Assistant',page_icon='📄',layout='wide')
st.title('Health Insurance Policy Assistant')
st.caption('Educational RAG demo based on PDFs in data/. Verify the wording and your own policy schedule.')
with st.sidebar:
    st.header('Pipeline: V4')
    st.write('FAISS retrieval → CrossEncoder reranking → Google Gemini')
    st.warning('Not an insurer service. This app cannot decide or guarantee claim approval.')
q=st.text_area('Enter your question',placeholder='How early should I notify the insurer for planned hospitalization?')
if st.button('Get answer',type='primary',disabled=not q.strip()):
    try:
        with st.spinner('Retrieving evidence and generating answer...'):
            start=time.perf_counter(); answer,sources,candidates,selected=ask(q.strip()); elapsed=time.perf_counter()-start
        st.subheader('Answer'); st.write(answer)
        st.caption(f'{elapsed:.2f} seconds | {len(candidates)} candidates → {len(selected)} reranked chunks')
        st.subheader('Source documents and pages')
        if sources:
            for i,s in enumerate(sources,1):
                with st.expander(f"{i}. {s['source']} — PDF page {s['page']} ({s['document_type']})"):
                    st.write(s['excerpt'])
        else: st.info('No source passages returned.')
    except Exception as e:
        st.error('Request failed. Check GOOGLE_API_KEY in .env, Gemini API/model access, dependencies, and that PDF files exist in data/.')
        st.exception(e)
