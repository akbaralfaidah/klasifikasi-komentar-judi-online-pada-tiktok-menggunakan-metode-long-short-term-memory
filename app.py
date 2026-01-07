import streamlit as st
import pandas as pd
import numpy as np
import os
import math
from preprocessing import Preprocessor
from word_embedding import WordEmbedding
from model_builder import ModelBuilder

# ==========================================
# 1. KONFIGURASI HALAMAN & VARIABEL GLOBAL
# ==========================================
st.set_page_config(
    page_title="Interface Klasifikasi Komentar Judi Online TikTok",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

TOKENIZER_PATH = 'tokenizer.json'
MODEL_DIR = '.' 
VALIDATION_DATA_MENTAH = 'data_validasi_mentah.csv' 

DATA_SKENARIO = {
    "model 1.h5":  {"lr": "0.001", "bs": 32, "epoch": 5},
    "model 2.h5":  {"lr": "0.001", "bs": 32, "epoch": 10},
    "model 3.h5":  {"lr": "0.001", "bs": 32, "epoch": 25},
    "model 4.h5":  {"lr": "0.001", "bs": 64, "epoch": 5},
    "model 5.h5":  {"lr": "0.001",  "bs": 64, "epoch": 10},
    "model 6.h5":  {"lr": "0.001",  "bs": 64, "epoch": 25},
    "model 7.h5":  {"lr": "0.0001",  "bs": 32, "epoch": 5},
    "model 8.h5":  {"lr": "0.0001",  "bs": 32, "epoch": 10},
    "model 9.h5":  {"lr": "0.0001", "bs": 32, "epoch": 25},
    "model 10.h5": {"lr": "0.0001", "bs": 64, "epoch": 50},
    "model 11.h5": {"lr": "0.0001", "bs": 64, "epoch": 10},
    "model 12.h5": {"lr": "0.0001", "bs": 64, "epoch": 25},
}

def inject_custom_css():
    BIRU_MUDA_AKTIF = "#0E256A"
    TEKS_PUTIH = "#FAFAFA"

    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: #0E1117; }}
            
            .block-container {{
                padding-top: 2rem !important;
                padding-bottom: 3rem !important;
                max-width: 95% !important;
            }}
            
            /* Tombol Primary */
            button.stButton:not([kind="secondary"]):not([kind="link"]):not([kind="minimal"]) {{
                background-color: {BIRU_MUDA_AKTIF} !important;
                color: {TEKS_PUTIH} !important;
                border: 1px solid {BIRU_MUDA_AKTIF} !important;
                font-weight: bold;
                border-radius: 8px;
                height: 3em;
            }}
            
            /* Container Styling */
            div[data-testid="stVerticalBlockBorderWrapper"] > div > div {{
                background-color: #161B22; 
                border-radius: 10px;
                padding: 20px;
                border: 1px solid #30363D;
            }}
            
            /* Judul Section (H3) */
            h3 {{
                color: #58A6FF;
                font-weight: 600;
                font-size: 1.1rem !important;
                margin-bottom: 0.5rem !important;
            }}
            
            /* --- FIX DROPDOWN SCROLL (PENTING) --- */
            /* Memaksa list dropdown (ul) memiliki tinggi maksimal 300px dan scrollbar aktif */
            div[role="listbox"] ul {{
                max-height: 300px !important;
                overflow-y: auto !important;
            }}
            
            /* Metrics Styling */
            [data-testid="stMetricLabel"] {{ font-size: 0.8rem; color: #8B949E; }}
            [data-testid="stMetricValue"] {{ font-size: 1.2rem; color: #FAFAFA; font-weight: 700; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ==========================================
# 3. FUNGSI HELPER
# ==========================================
def determine_category(row):
    actual = row['label']
    pred = row['prediksi_biner']
    if actual == 1 and pred == 1: return "TP (True Positive)"
    elif actual == 0 and pred == 0: return "TN (True Negative)"
    elif actual == 0 and pred == 1: return "FP (False Positive)"
    elif actual == 1 and pred == 0: return "FN (False Negative)"
    return "Unknown"

@st.cache_resource
def load_preprocessor():
    return Preprocessor()

@st.cache_resource
def load_word_embedding(tokenizer_path):
    we = WordEmbedding()
    try:
        we.load_tokenizer(tokenizer_path)
    except Exception as e:
        st.error(f"FATAL: Gagal memuat {tokenizer_path}. Error: {e}")
        st.stop()
    return we

@st.cache_resource
def load_and_process_validation_data(_preprocessor, filepath, _progress_bar, _status_text):
    try:
        df_raw = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"FATAL: File `{filepath}` tidak ditemukan.")
        st.stop()
    if 'text' not in df_raw.columns:
        st.error(f"FATAL: File `{filepath}` tidak memiliki kolom 'text'.")
        st.stop()
        
    total = len(df_raw)
    processed_texts = []
    for i, text in enumerate(df_raw['text']):
        progress_percent = (i + 1) / total
        _progress_bar.progress(progress_percent)
        _status_text.text(f"Memproses data validasi: {i + 1}/{total} baris...")
        processed_texts.append(" ".join(_preprocessor.preprocess_text(text)))
        
    df_processed = df_raw.copy()
    df_processed['processed_text'] = processed_texts
    return df_processed

def process_dataframe(df_raw, preprocessor):
    processed_texts = []
    progress_bar = st.progress(0, text="Memproses file (Sastrawi)...")
    total = len(df_raw)
    for i, text in enumerate(df_raw['text']):
        processed_texts.append(" ".join(preprocessor.preprocess_text(text)))
        progress_bar.progress((i + 1) / total, text=f"Memproses baris {i+1}/{total}")
    progress_bar.empty()
    df_processed = df_raw.copy()
    df_processed['processed_text'] = processed_texts
    return df_processed

def show_paginated_results(df_results, state_key):
    st.subheader("📊 Hasil Klasifikasi")
    
    df_display_final = df_results.copy()
    
    if 'kategori_evaluasi' in df_results.columns:
        col_filter, col_item = st.columns([3, 1])
        with col_filter:
            filter_options = ["TP (True Positive)", "TN (True Negative)", "FP (False Positive)", "FN (False Negative)"]
            selected_filters = st.multiselect(
                "Filter Kategori Evaluasi:",
                options=filter_options,
                default=[],
                key=f"{state_key}_filter",
                on_change=lambda: st.session_state.update({f"{state_key}_page": 1})
            )
            if selected_filters:
                df_display_final = df_results[df_results['kategori_evaluasi'].isin(selected_filters)]
            else:
                df_display_final = df_results
        with col_item:
            items_per_page = st.selectbox("Item per Halaman", (10, 20, 50, 100), key=f"{state_key}_items", on_change=lambda: st.session_state.update({f"{state_key}_page": 1}))
    else:
        col_dummy, col_item = st.columns([3, 1])
        with col_item:
            items_per_page = st.selectbox("Item per Halaman", (10, 20, 50, 100), key=f"{state_key}_items", on_change=lambda: st.session_state.update({f"{state_key}_page": 1}))

    total_items = len(df_display_final)
    total_pages = max(1, math.ceil(total_items / items_per_page))
    st.caption(f"Menampilkan {total_items} data.")

    page_state_key = f"{state_key}_page"
    if page_state_key not in st.session_state:
        st.session_state[page_state_key] = 1
    if st.session_state[page_state_key] > total_pages:
        st.session_state[page_state_key] = total_pages

    current_page = st.session_state[page_state_key]
    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    
    df_page_view = df_display_final.iloc[start_idx:end_idx].copy()
    df_page_view.insert(0, 'No.', range(start_idx + 1, start_idx + 1 + len(df_page_view)))
    df_page_view['skor_prediksi'] = df_page_view['skor_prediksi'].apply(lambda x: f"{x:.4f}")
    
    cols_to_show = ['No.', 'text', 'skor_prediksi', 'klasifikasi']
    if 'kategori_evaluasi' in df_page_view.columns:
        cols_to_show.append('kategori_evaluasi')
        
    st.dataframe(
        df_page_view[cols_to_show],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    PAGE_WINDOW_SIZE = 5 
    mid_point = PAGE_WINDOW_SIZE // 2
    start_page = max(1, current_page - mid_point)
    end_page = min(total_pages, start_page + PAGE_WINDOW_SIZE - 1)
    if end_page == total_pages:
        start_page = max(1, total_pages - PAGE_WINDOW_SIZE + 1)
    
    buttons_to_show = [{"label": "⏮", "page": current_page - 1, "disabled": current_page == 1, "key_suffix": "prev"}]
    for page_num in range(start_page, end_page + 1):
        buttons_to_show.append({"label": str(page_num), "page": page_num, "is_active": current_page == page_num, "key_suffix": f"page_{page_num}"})
    buttons_to_show.append({"label": "⏭", "page": current_page + 1, "disabled": current_page == total_pages, "key_suffix": "next"})

    spacer_left, center_col, spacer_right = st.columns([1, 2, 1]) 
    with center_col:
        cols = st.columns(len(buttons_to_show))
        for i, button_info in enumerate(buttons_to_show):
            with cols[i]:
                btn_type = "primary" if button_info.get("is_active", False) else "secondary"
                st.button(
                    button_info["label"],
                    key=f"{state_key}_btn_{button_info['key_suffix']}",
                    type=btn_type,
                    disabled=button_info.get("disabled", False),
                    on_click=lambda p=button_info['page']: st.session_state.update({page_state_key: p}),
                    use_container_width=True
                )

# ==========================================
# 4. MAIN PROGRAM
# ==========================================
def main():
    inject_custom_css()
    
    with st.spinner("Menyiapkan sistem..."):
        preprocessor = load_preprocessor()
        word_embedding = load_word_embedding(TOKENIZER_PATH)
    
    if 'model_builder' not in st.session_state:
        st.session_state.model_builder = ModelBuilder(preprocessor, word_embedding)
    if 'active_model_name' not in st.session_state:
        st.session_state.active_model_name = None
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    if 'show_table' not in st.session_state:
        st.session_state.show_table = False

    if 'df_validasi_processed' not in st.session_state:
        loading_container = st.empty()
        with loading_container.container():
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.title("Inisialisasi Aplikasi")
            st.write("Mohon tunggu, sistem sedang memproses data validasi awal... 🚀")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            df_processed = load_and_process_validation_data(
                _preprocessor=preprocessor, 
                filepath=VALIDATION_DATA_MENTAH, 
                _progress_bar=progress_bar, 
                _status_text=status_text
            )
            st.session_state.df_validasi_processed = df_processed
        loading_container.empty()
        st.rerun()

    df_validasi_processed = st.session_state.df_validasi_processed
    
    st.markdown(
        """
        <h1 style='text-align: center; font-weight: 800; font-size: 30px; margin-bottom: 30px;'>
            KLASIFIKASI KOMENTAR JUDI ONLINE PADA TIKTOK <br>
            MENGGUNAKAN METODE LONG SHORT-TERM MEMORY
        </h1>
        """,
        unsafe_allow_html=True
    )
    
    tab1, tab2 = st.tabs(["Pilih Model Skenario", "Klasifikasi Teks Baru"])
    
    # --- TAB 1: PILIH MODEL ---
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)

        col_kiri, col_kanan = st.columns([1.5, 1], gap="medium")

        with col_kiri:
            with st.container(border=True):
                st.markdown("### 🛠️ Konfigurasi Model")
                st.write("Pilih salah satu dari 12 skenario model untuk dimuat.")
                
                dropdown_options = []
                option_to_filename = {} 

                for i, (filename, params) in enumerate(DATA_SKENARIO.items()):
                    label = f"Model {i+1} (LR: {params['lr']} | Batch: {params['bs']} | Epoch: {params['epoch']})"
                    if filename == "model 6.h5":
                        label += " 🥇 [PERINGKAT 1]"
                    elif filename == "model 5.h5":
                        label += " 🥈 [PERINGKAT 2]"
                    elif filename == "model 2.h5":
                        label += " 🥉 [PERINGKAT 3]"
                    dropdown_options.append(label)
                    option_to_filename[label] = filename

                selected_label = st.selectbox(
                    "Pilih Skenario Pengujian:",
                    options=dropdown_options,
                    index=0
                )
                
                target_filename = option_to_filename[selected_label]
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🚀 Muat & Aktifkan Model", type="primary", use_container_width=True):
                    with st.spinner(f"Sedang memuat {target_filename}..."):
                        success = st.session_state.model_builder.load_model(
                            os.path.join(MODEL_DIR, target_filename)
                        )
                        
                        if success:
                            st.session_state.active_model_name = target_filename
                            scores = st.session_state.model_builder.classify_batch(
                                df_validasi_processed['processed_text']
                            )
                            df_results = df_validasi_processed.copy()
                            df_results['skor_prediksi'] = scores
                            df_results['prediksi_biner'] = [1 if s >= 0.5 else 0 for s in scores]
                            df_results['klasifikasi'] = [
                                "Judi Online" if s == 1 else "Non-Judi Online" for s in df_results['prediksi_biner']
                            ]
                            if 'label' in df_results.columns:
                                df_results['kategori_evaluasi'] = df_results.apply(determine_category, axis=1)

                            st.session_state.validation_results = df_results
                            st.session_state.show_table = True 
                            st.rerun()
                        else:
                            st.error(f"Gagal memuat file: {target_filename}")

        with col_kanan:
            with st.container(border=True):
                st.markdown("### Status Model")
                st.write("Informasi model yang sedang aktif.")

                if st.session_state.active_model_name:  
                    active_name = st.session_state.active_model_name
                    current_conf = DATA_SKENARIO.get(active_name, {})
                    
                    st.success(f"**AKTIF:** {active_name}")
                    
                    # --- LOGIKA TAMBAHAN UNTUK PERINGKAT ---
                    rank_info = None
                    if active_name == "model 6.h5":
                        rank_info = {"msg": "🏆 PERINGKAT 1 (TERBAIK)", "color": "#FFD700"} # Emas
                    elif active_name == "model 5.h5":
                        rank_info = {"msg": "🥈 PERINGKAT 2", "color": "#C0C0C0"} # Perak
                    elif active_name == "model 2.h5":
                        rank_info = {"msg": "🥉 PERINGKAT 3", "color": "#CD7F32"} # Perunggu
                    
                    # Jika model termasuk 3 besar, tampilkan badge khusus
                    if rank_info:
                        st.markdown(
                            f"""
                            <div style="
                                background-color: rgba(255, 255, 255, 0.05); 
                                border: 1px solid {rank_info['color']};
                                border-left: 6px solid {rank_info['color']}; 
                                padding: 12px; 
                                margin-top: 10px; 
                                margin-bottom: 10px; 
                                border-radius: 6px;">
                                <h4 style="color: {rank_info['color']}; margin:0; padding:0; font-size: 16px;">
                                    {rank_info['msg']}
                                </h4>
                                <span style="font-size: 12px; color: #ccc;">
                                    Model ini memiliki performa akurasi tinggi.
                                </span>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    # ---------------------------------------

                    st.divider()
                    
                    m1, m2, m3 = st.columns(3)
                    with m1: st.metric("L.Rate", current_conf.get('lr', '-'))
                    with m2: st.metric("Batch", current_conf.get('bs', '-'))
                    with m3: st.metric("Epoch", current_conf.get('epoch', '-'))
                else:
                    st.markdown(
                        """
                        <div style="background-color: #262730; padding: 15px; border-radius: 5px; text-align: center; color: #FAFAFA; border: 1px dashed #58A6FF;">
                            ⚠️ <strong>BELUM ADA MODEL AKTIF</strong> <br><br>
                            Silakan pilih skenario di panel kiri <br>
                            lalu klik tombol <strong>Muat & Validasi</strong>.
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("---")
        if st.session_state.show_table and st.session_state.validation_results is not None:
             show_paginated_results(st.session_state.validation_results, state_key='val')

    # --- TAB 2: KLASIFIKASI BARU ---
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📝 Uji Coba Klasifikasi")

        if st.session_state.active_model_name is None:
            st.error("⚠️ Anda belum memilih model. Silakan kembali ke tab 'Pilih Model Skenario'.")
        else:
            col_input1, col_input2 = st.columns(2, gap="medium")

            with col_input1:
                with st.container(border=True):
                    st.subheader("1. Teks Tunggal")
                    text_input = st.text_area("Masukkan komentar:", height=150, placeholder="Contoh: Info gacor...")
                    
                    if st.button("🔍 Check Komentar", type="primary", use_container_width=True):
                        if text_input:
                            with st.spinner("Memproses..."):
                                score = st.session_state.model_builder.classify_text(text_input)
                            st.markdown("#### Hasil Prediksi:")
                            if score >= 0.5:
                                st.error(f"🛑 **JUDI ONLINE**\n\nPrediksi: {score:.4f}")
                            else:
                                st.success(f"✅ **NON-JUDI**\n\nPrediksi: {score:.4f}")
                        else:
                            st.warning("Isi teks komentar dahulu.")

            with col_input2:
                with st.container(border=True):
                    st.subheader("2. Upload File (Batch)")
                    uploaded_file_new = st.file_uploader("Upload .csv/.txt", type=['csv', 'txt'])
                    
                    if uploaded_file_new:
                        if st.button("📂 Check File", type="primary", use_container_width=True):
                            try:
                                if uploaded_file_new.name.endswith('.csv'):
                                    df_new_raw = pd.read_csv(uploaded_file_new)
                                else:
                                    lines = uploaded_file_new.getvalue().decode('utf-8').splitlines()
                                    df_new_raw = pd.DataFrame(lines, columns=['text'])
                                
                                df_new_processed = process_dataframe(df_new_raw, preprocessor)
                                scores_new = st.session_state.model_builder.classify_batch(df_new_processed['processed_text'])
                                df_results_new = df_new_processed.copy()
                                df_results_new['skor_prediksi'] = scores_new
                                df_results_new['klasifikasi'] = ["Judi Online" if s >= 0.5 else "Non-Judi Online" for s in scores_new]
                                st.session_state.df_file_processed = df_results_new
                            except Exception as e:
                                st.error(f"Error: {e}")

            if 'df_file_processed' in st.session_state and st.session_state.df_file_processed is not None:
                 st.markdown("---")
                 show_paginated_results(st.session_state.df_file_processed, state_key='file')

if __name__ == "__main__":
    main()