import streamlit as st
from function import predict

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def app(data, x, y):
    st.title("HALAMAN PREDIKSI")

    columns = st.columns(4)

    # Kolom 1
    with columns[0]:
        age = st.text_input('Input Usia')
        bp = st.text_input('Input Nilai bp (Blood Pressure)')
        sg = st.text_input('Input Nilai sg (Specific Gravity)')
        al = st.text_input('Input Nilai al (Albumin)')
        su = st.text_input('Input Nilai su (Sugar)')
        rbc = st.text_input('Input Nilai rbc (Red Blood Cells)')
        pc = st.text_input('Input Nilai pc (Pus Cell)')
    
    # Kolom 2
    with columns[1]:
        pcc = st.text_input('Input Nilai pcc (Pus Cell clumps)')
        ba = st.text_input('Input Nilai ba (Bacteria)')
        bgr = st.text_input('Input Nilai bgr (Blood Glucose Random)')
        bu = st.text_input('Input Nilai bu (Blood Urea)')
        sc = st.text_input('Input Nilai sc (Serum Creatinine)')
        sod = st.text_input('Input Nilai sod (Sodium)')
        pot = st.text_input('Input Nilai pot (Potassium)')
    
    # Kolom 3
    with columns[2]:
        hemo = st.text_input('Input Nilai hemo (Hemoglobin)')
        pcv = st.text_input('Input Nilai pcv (Packed Cell Volume)')
        wc = st.text_input('Input Nilai wc (White Blood Cell Count)')
        rc = st.text_input('Input Nilai rc (Red Blood Cell Count)')
        htn = st.text_input('Input Nilai htn (Hypertension)')
    
    # Kolom 4
    with columns[3]:
        dm = st.text_input('Input Nilai dm (Diabetes Mellitus)')
        cad = st.text_input('Input Nilai cad (Coronary Artery Disease)')
        appet = st.text_input('Input Nilai appet (Appetite)')
        pe = st.text_input('Input Nilai pe (Pedal Edema)')
        ane = st.text_input('Input Nilai ane (Anemia)')

    # Deklarasi variabel features setelah semua variabel telah didefinisikan
    features = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]

    if st.button("Prediksi"):
        # Periksa input yang kosong
        if any(feature == '' for feature in features):
            st.warning("Harap lengkapi semua input sebelum melakukan prediksi!")
        # Periksa input yang bukan numerik
        elif not all(is_numeric(feature) for feature in features if feature != ''):
            st.warning("Harap masukkan nilai numerik untuk melakukan prediksi!")
        else:
            prediction, score = predict(x, y, features)
            score = score
            st.info("Prediksi Sukses")
            if prediction == 1:
                st.warning("Orang Tersebut Dapat Terkena Ginjal")
            else:
                st.success("Orang Tersebut Aman Terkena Ginjal")
                st.write("Model yang digunakan memiliki tingkat Akurasi", (score * 100), "%")
