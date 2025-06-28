import streamlit as st
st.set_page_config(layout="wide")
st.title("Hệ thống Phân tích & Đề xuất Doanh nghiệp IT")

# --- CSS để cố định thông tin lớp/học viên ở cuối sidebar ---
st.markdown("""
<style>
    .st-emotion-cache-vk3305 { /* Đây là class của sidebar chính */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Đẩy nội dung lên trên và xuống dưới */
    }
    .fixed-bottom-left {
        position: sticky; /* Hoặc fixed nếu bạn muốn nó luôn ở đó ngay cả khi cuộn */
        bottom: 0;
        left: 0; /* Đảm bảo nó nằm sát mép trái của sidebar */
        width: 100%; /* Chiếm toàn bộ chiều rộng của sidebar */
        padding: 1rem; /* Thêm padding cho đẹp */
        background-color: #f0f2f6; /* Màu nền giống sidebar hoặc màu bạn muốn */
        border-top: 1px solid #e0e0e0; /* Đường viền phía trên để tách biệt */
        box-sizing: border-box; /* Đảm bảo padding không làm tăng kích thước */
        font-size: 0.95rem;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Logo, Menu và Thông tin học viên ---
with st.sidebar:
    # 0. Logo nhỏ ở đầu sidebar
    st.image("img\your_logo.jpg", width=60)
    # 1. Tiêu đề sidebar ngắn gọn
    st.markdown("<h3 style='margin-bottom:0.5rem;'>IT Recommendation System</h3>", unsafe_allow_html=True)
    # 2. Thanh select box
    menu_selection = st.selectbox(
        "Menu",
        ["Business Problem", "Build Project", "New Prediction"]
    )
    # 3. Thông tin học viên thực hiện (cố định dưới cùng, rút gọn, có link email)
    st.markdown(
        """
        <div class="fixed-bottom-left" style="background-color:#f0f2f6; border-top:1px solid #e0e0e0;">
        <b>LỚP DL07_K304 - DATA SCIENCE - MACHINE LEARNING</b><br>
        Học viên thực hiện:<br>
        - <b>Ms. Giang Phi Yến</b> - <a href='mailto:yengp96@gmail.com'>Email</a><br>
        - <b>Ms. Nguyễn Ngọc Khánh Linh</b> - <a href='mailto:nnkl1517000@gmail.com'>Email</a>
        </div>
        """,
        unsafe_allow_html=True
    )
if menu_selection == "Business Problem":
    st.header("Hiểu rõ Vấn đề Kinh doanh")
    st.markdown("""
    Ứng dụng này nhằm giải quyết hai vấn đề cốt lõi trong lĩnh vực tuyển dụng và đánh giá doanh nghiệp IT tại Việt Nam.
    Chúng tôi sử dụng dữ liệu từ nền tảng tuyển dụng ITviec để cung cấp các phân tích và dự đoán hữu ích.
    """)

    st.subheader("1.1. Gợi ý các Công ty Tương tự")
    st.markdown("""
    * **Nguồn dữ liệu:** Dựa trên nội dung mô tả công ty, vị trí tuyển dụng và các thông tin liên quan từ **ITviec.com**.
    * **Mục tiêu:** Đề xuất các công ty có hồ sơ kinh doanh và yêu cầu tuyển dụng tương đồng. Việc này giúp các nhà tuyển dụng tìm kiếm đối thủ cạnh tranh, các ứng viên khám phá thêm cơ hội việc làm phù hợp, hoặc các bên liên quan học hỏi từ mô hình hoạt động của các công ty khác.
    """)
    st.info("💡 Bạn có thể trải nghiệm tính năng này tại mục 'New Prediction' sau khi mô hình được xây dựng.")

    st.subheader("1.2. Dự đoán khả năng 'Recommend' (Đề xuất) của Công ty")
    st.markdown("""
    * **Nguồn dữ liệu:** Dựa trên các bài đánh giá, nhận xét của nhân viên và ứng viên đã làm việc hoặc phỏng vấn tại các công ty.
    * **Mục tiêu:** Dự đoán liệu một công ty cụ thể có nhận được đánh giá tích cực và được đề xuất bởi nhân viên/ứng viên hay không. Điều này hỗ trợ các ứng viên đưa ra quyết định thông minh hơn khi tìm việc và giúp các công ty nhận diện điểm mạnh/yếu của mình.
    """)
    st.info("💡 Bạn có thể trải nghiệm tính năng này tại mục 'New Prediction' sau khi mô hình được xây dựng.")

elif menu_selection == "Build Project":
    st.header("Xây dựng & Huấn luyện Mô hình")
    tabs_build = st.tabs([
        "Preprocessing",
        "Content-based Company Suggestion",
        "Classification for Recommend"
    ])

    # Tab 0: Preprocessing
    with tabs_build[0]:
        st.subheader("Tiền xử lý dữ liệu (Preprocessing)")
        st.markdown("""
        - Làm sạch văn bản, chuẩn hóa tiếng Việt, loại bỏ stopwords, ký tự đặc biệt.
        - Tách từ, chuyển đổi về dạng số (vector hóa) phục vụ cho các mô hình học máy.
        - Chuẩn hóa các đặc trưng số, xử lý dữ liệu thiếu và mã hóa nhãn nếu cần.
        - Tiền xử lý là bước quan trọng giúp tăng chất lượng đầu vào cho mô hình.
        """)
        img_paths_pre = [
            "img/Project2_Img (5).PNG",
            "img/Project2_Img (6).PNG",
            "img/Project2_Img (7).PNG",
            "img/Project2_Img (8).PNG"
        ]
        for img_path in img_paths_pre:
            st.image(img_path, use_container_width=True)

    # Tab 1: Gợi ý công ty tương tự
    with tabs_build[1]:
        st.subheader("Quy trình & Kết quả: Gợi ý công ty tương tự")
        img_paths_1 = [
            "img/Project2_Img (10).PNG",
            "img/Project2_Img (11).PNG",
            "img/Project2_Img (12).PNG",
            "img/Project2_Img (13).PNG",
            "img/Project2_Img (14).PNG",
            "img/Project2_Img (15).PNG"
        ]
        for img_path in img_paths_1:
            st.image(img_path, use_container_width=True)

    # Tab 2: Dự đoán Recommend
    with tabs_build[2]:
        st.subheader("Quy trình & Kết quả: Dự đoán Recommend")
        img_paths_2 = [
            "img/Project2_Img (17).PNG",
            "img/Project2_Img (18).PNG",
            "img/Project2_Img (19).PNG",
            "img/Project2_Img (20).PNG",
            "img/Project2_Img (21).PNG",
            "img/Project2_Img (22).PNG",
            "img/Project2_Img (23).PNG"
        ]
        for img_path in img_paths_2:
            st.image(img_path, use_container_width=True)

elif menu_selection == "New Prediction":
    st.header("Dự đoán Mới & Gợi ý")
    st.write("Sử dụng các mô hình đã huấn luyện để đưa ra dự đoán và gợi ý.")

    tabs = st.tabs(["Gợi ý công ty từ mô tả người dùng", "Dự đoán Recommend"])

    # --- Tab 1: Gợi ý công ty từ mô tả người dùng ---
    with tabs[0]:
        st.header("Gợi ý công ty từ mô tả người dùng")
        import pandas as pd
        try:
            df_companies = pd.read_excel("data/Overview_Companies_ongoing.xlsx")
            company_names = df_companies['Company Name'].dropna().unique().tolist()
            selected_company = st.selectbox("Chọn công ty", company_names)
            company_info = df_companies[df_companies['Company Name'] == selected_company].iloc[0]
            st.markdown(f"""
            ### {company_info['Company Name']}
            **Industry:** {company_info.get('Company industry', 'N/A')}
            
            **Information:**
            {company_info.get('Company overview', 'N/A')}
            
            **Key skills:**
            {company_info.get('Our key skills', 'N/A')}
            """)
            # Hiển thị các công ty liên quan (ví dụ: cùng ngành)
            related_companies = df_companies[(df_companies['Company industry'] == company_info['Company industry']) & (df_companies['Company Name'] != selected_company)]
            st.markdown("**Các công ty liên quan:**")
            cols = st.columns(min(3, len(related_companies)))
            for idx, (_, row) in enumerate(related_companies.head(3).iterrows()):
                with cols[idx]:
                    st.markdown(f"**{row['Company Name']}**")
                    st.caption(row.get('Company overview', '')[:150] + '...')
                    st.caption(f"Key skills: {row.get('Our key skills', '')}")
        except Exception as e:
            st.error(f"Không thể tải dữ liệu công ty: {e}")
        st.markdown("---")
        st.subheader("Phân nhóm công ty bằng KMeans")
        n_clusters = st.slider("Số nhóm (cluster)", min_value=2, max_value=10, value=5)
        if st.button("Phân nhóm công ty", key="cluster_btn"):
            try:
                from utils.company_suggestion import preprocess_company_text, build_gensim_tfidf, cluster_companies_kmeans
                df_companies = preprocess_company_text(df_companies, 'Company overview')
                dictionary, corpus, tfidf, corpus_tfidf = build_gensim_tfidf(df_companies, 'Company overview')
                labels, kmeans = cluster_companies_kmeans(corpus_tfidf, n_clusters=n_clusters)
                df_companies['Cluster'] = labels
                st.write(df_companies[['Company Name', 'Cluster']])
            except Exception as e:
                st.error(f"Lỗi khi phân nhóm: {e}")

    # --- Tab 2: Dự đoán Recommend ---
    with tabs[1]:
        st.header("Dự đoán Recommend cho công ty")
        with st.expander("Nhập range mong muốn cho từng tiêu chí (1-5)"):
            st.markdown("<style>div[data-baseweb='slider'] .css-1n76uvr, div[data-baseweb='slider'] .stSlider {background: #e0e0e0 !important;}</style>", unsafe_allow_html=True)
            salary = st.slider("Salary & benefits", 1, 5, 3, key="salary_slider")
            training = st.slider("Training & learning", 1, 5, 3, key="training_slider")
            management = st.slider("Management cares about me", 1, 5, 3, key="management_slider")
            culture = st.slider("Culture & fun", 1, 5, 3, key="culture_slider")
            office = st.slider("Office & workspace", 1, 5, 3, key="office_slider")
        review_content = st.text_area("Nhập nội dung review (bằng tiếng Anh):")
        if st.button("Dự đoán Recommend", key="recommend_btn"):
            import time
            with st.spinner("Đang xử lý..."):
                try:
                    import pickle
                    from scipy.sparse import hstack
                    with open("output/recommend_rf.pkl", "rb") as f:
                        pipeline = pickle.load(f)
                    model = pipeline['model']
                    vectorizer = pipeline['vectorizer']
                    scaler = pipeline['scaler']
                    X_text = vectorizer.transform([review_content])
                    X_num = scaler.transform([[salary, training, management, culture, office]])
                    X_input = hstack([X_text, X_num])
                    pred = model.predict(X_input)[0]
                    proba = model.predict_proba(X_input)[0, 1]
                    if pred == 1:
                        st.success(f"✅ Recommend (Xác suất: {proba:.2%})")
                        st.metric("Kết quả", "Recommend", delta=f"{proba:.2%}")
                    else:
                        st.warning(f"❌ Not Recommend (Xác suất: {proba:.2%})")
                        st.metric("Kết quả", "Not Recommend", delta=f"{proba:.2%}")
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {e}")
