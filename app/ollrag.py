import streamlit as st

st.markdown(
    """
    <style>
    .stMainBlockContainer{
        padding: 1rem 1rem 10rem;
    }
    .custom-text {
        font-size: 12px;
    }
    [data-testid="stDecoration"] {
		display: none;
	}
    """,
    unsafe_allow_html=True,
)

pages = {
    "Pages": [
        st.Page("chat_page.py", title="Chat", icon="💬"),
        st.Page("data_page.py", title="Data", icon="📂"),
    ],
    "Chats": [],
}

pg = st.navigation(pages)
pg.run()
