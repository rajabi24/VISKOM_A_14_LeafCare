"""
disease_info.py - Informasi lengkap dan rekomendasi agronomis untuk penyakit daun cabai.
Mencakup data patogen, gejala, tingkat risiko, penanganan kimiawi & hayati, serta pencegahan.
Bebas dari emoji kartun untuk menjaga estetika profesional editorial.
"""

DISEASE_DATABASE = {
    "Bacterial Spot": {
        "indonesian_name": "Bercak Bakteri",
        "scientific_name": "Xanthomonas campestris pv. vesicatoria",
        "category": "Bakteri (Bacterial)",
        "severity": "Tinggi",
        "severity_color": "#DC2626",  # Red
        "icon_name": "shield",
        "summary": "Penyakit bercak bakteri sangat menular terutama pada musim hujan atau kondisi kelembapan tinggi, menyebabkan gugur daun dini dan penurunan hasil panen secara drastis.",
        "symptoms": [
            "Bercak kecil melingkar atau bersegi yang tampak basah/berair (water-soaked) pada daun muda.",
            "Bercak berkembang menjadi cokelat tua hingga hitam keunguan dengan lingkaran halo kuning di sekitarnya.",
            "Daun yang terinfeksi parah akan menguning dan rontok (defoliasi) sebelum waktunya.",
            "Pada buah juga dapat muncul lentisel kasar bercelak cokelat menyerupai kerak."
        ],
        "cause": "Infeksi bakteri Xanthomonas campestris pv. vesicatoria yang menyebar melalui percikan air hujan, alat pertanian yang terkontaminasi, dan benih yang terinfeksi.",
        "chemical_control": [
            "Semprotkan bakterisida berbahan aktif Tembaga Hidroksida (Kocide 54WDG) atau Tembaga Oksiklorida (Cupravit).",
            "Gunakan antibiotik tanaman berbahan aktif Kasugamisin (Kasumin 20SL) atau Streptomisin Sulfat sesuai dosis anjuran.",
            "Lakukan rotasi penyemprotan setiap 5-7 hari sekali saat cuaca lembap atau musim hujan."
        ],
        "organic_control": [
            "Aplikasi agen hayati perakaran dan daun dengan Bacillus subtilis atau Pseudomonas fluorescens.",
            "Semprotkan biopestisida nabati dari ekstrak rebusan daun sirih merah, kunyit, dan daun mimba.",
            "Gunakan larutan PGPR (Plant Growth-Promoting Rhizobacteria) untuk meningkatkan ketahanan sistemik tanaman."
        ],
        "prevention": [
            "Gunakan benih cabai bersertifikat yang bebas penyakit dan tahan patogen.",
            "Hindari sistem pengairan siram tajuk (overhead sprinkler); gunakan irigasi tetes atau siram langsung ke zona akar.",
            "Buang dan bakar daun atau tanaman yang terinfeksi agar tidak menular ke tanaman lain.",
            "Jaga drainase bedengan dan pasang mulsa plastik perak untuk mengurangi kelembapan tanah."
        ]
    },
    "Cercospora Leaf Spot": {
        "indonesian_name": "Bercak Daun Cercospora (Mata Katak)",
        "scientific_name": "Cercospora capsici Heald & Wolf",
        "category": "Jamur / Fungi",
        "severity": "Sedang",
        "severity_color": "#D97706",  # Amber
        "icon_name": "microscope",
        "summary": "Dikenal sebagai penyakit 'Mata Katak' (Frog-Eye Leaf Spot), umum terjadi pada kondisi hangat dan lembap, menyerang permukaan daun dan tangkai.",
        "symptoms": [
            "Bercak melingkar berukuran 0.5 - 1 cm dengan titik tengah berwarna abu-abu keputihan.",
            "Tepi bercak berwarna cokelat kemerahan atau cokelat gelap yang tegas menyerupai pupil mata katak.",
            "Jika bercak semakin banyak, daun menguning di sekitarnya kemudian layu dan berguguran.",
            "Batang dan tangkai bunga juga dapat terserang menyebabkan bunga rontok."
        ],
        "cause": "Jamur Cercospora capsici yang spora konidianya disebarkan oleh angin dan percikan air dalam kondisi kelembapan relatif tinggi (>80%) dan suhu 25-30°C.",
        "chemical_control": [
            "Aplikasikan fungisida kontak berbahan aktif Mankozeb (Dithane M-45) atau Klorotalonil (Daconil).",
            "Gunakan fungisida sistemik berbahan aktif Difenokonazol (Score 250EC) atau Azoksistrobin (Amistar Top).",
            "Semprotkan dengan interval 7-10 hari sekali pada pagi hari."
        ],
        "organic_control": [
            "Semprotkan ekstrak daun nimba (Azadirachta indica) secara berkala.",
            "Aplikasi suspensi jamur antagonis Trichoderma harzianum pada tanaman dan media tanam.",
            "Gunakan fungisida hayati belerang organik (Sulfur terlarut) dosis rendah."
        ],
        "prevention": [
            "Atur jarak tanam ideal (minimal 50-60 cm) agar sirkulasi udara di bawah tajuk tanaman lancar.",
            "Pangkas wiwilan (tunas air) dan daun-daun tua bagian bawah yang dekat dengan tanah.",
            "Lakukan rotasi tanaman dengan komoditas bukan keluarga Solanaceae (bukan tomat, terung, tembakau).",
            "Bersihkan gulma di sekitar bedengan untuk menekan kelembapan mikro."
        ]
    },
    "Curl Virus": {
        "indonesian_name": "Virus Keriting (Gemini Virus / Bulai)",
        "scientific_name": "Chilli Leaf Curl Virus (ChiLCV) / Begomovirus",
        "category": "Virus & Vektor Hama",
        "severity": "Sangat Tinggi",
        "severity_color": "#991B1B",  # Dark Red
        "icon_name": "alert-triangle",
        "summary": "Penyakit virus yang sangat merusak tanaman cabai, ditularkan oleh vektor kutu kebul (Bemisia tabaci). Tanaman yang terserang dini seringkali tidak mampu menghasilkan buah sama sekali.",
        "symptoms": [
            "Daun muda menggulung atau mengeriting ke arah atas (cupping) atau ke arah bawah.",
            "Helai daun mengecil, menebal, kaku, dan terasa rapuh saat diremas.",
            "Tulang daun menebal dan menguning (vein clearing), warna daun belang kuning-hijau (mosaik).",
            "Pertumbuhan tanaman terhambat (kerdil / stunting) dengan ruas cabang memendek.",
            "Bunga mudah rontok dan buah yang terbentuk berukuran kecil, keras, dan cacat."
        ],
        "cause": "Infeksi virus Begomovirus yang ditularkan oleh serangga vektor kutu kebul (Bemisia tabaci) dan thrips. Virus tidak dapat disembuhkan dengan fungisida/bakterisida konvensional.",
        "chemical_control": [
            "Fokus kendalikan serangga vektor (kutu kebul) menggunakan insektisida sistemik berbahan aktif Imidakloprid (Confidor) atau Tiametokam (Actara).",
            "Gunakan akarisida/insektisida Abamektin (Demolish) atau Spinetoram bila terindikasi serangan thrips dan kutu.",
            "Semprotkan pupuk mikro zink (Zn) dan silika cair untuk memperkuat dinding sel tanaman yang masih sehat."
        ],
        "organic_control": [
            "Pasang perangkap likat kuning (Yellow Sticky Trap) sebanyak 40-50 lembar per hektar untuk menangkap kutu kebul dewasa.",
            "Semprotkan biopestisida minyak mimba (Neem Oil) dicampur sabun kalium sebagai perekat dan penembus kutu.",
            "Lepaskan musuh alami predator seperti larva Menochilus sexmaculatus (kumbang helm)."
        ],
        "prevention": [
            "Eradikasi (cabut dan musnahkan) tanaman yang menunjukkan gejala awal agar tidak menjadi sumber inokulum.",
            "Gunakan mulsa plastik hitam perak (MPHP) untuk memantulkan sinar UV dan mengusir kutu kebul.",
            "Tanam tanaman perangkap (border crops) seperti jagung atau tagetes di sekeliling lahan cabai.",
            "Pilih varietas cabai hibrida yang memiliki toleransi tinggi terhadap Gemini Virus."
        ]
    },
    "Healthy Leaf": {
        "indonesian_name": "Daun Sehat",
        "scientific_name": "Capsicum annuum L. (Kondisi Prima)",
        "category": "Normal / Sehat",
        "severity": "Aman",
        "severity_color": "#16A34A",  # Emerald Green
        "icon_name": "check-circle",
        "summary": "Tanaman cabai dalam kondisi fisiologis prima, bebas dari infeksi patogen jamur, bakteri, maupun serangan virus keriting.",
        "symptoms": [
            "Warna daun hijau segar merata tanpa bercak noda, nekrosis, atau klorosis.",
            "Permukaan daun mulus, elastis, dan bentukan helai daun terbuka sempurna.",
            "Susunan tulang daun normal dan tidak ada pembengkakan atau penebalan abnormal.",
            "Pertumbuhan vegetatif dan generatif berlangsung seimbang dan optimal."
        ],
        "cause": "Nutrisi tercukupi, kelembapan seimbang, sanitasi lahan terjaga, dan manajemen hama penyakit berjalan dengan baik.",
        "chemical_control": [
            "Tidak memerlukan aplikasi pestisida kuratif.",
            "Tetap lakukan pemupukan berimbang NPK sesuai fase pertumbuhan tanaman (vegetatif vs generatif)."
        ],
        "organic_control": [
            "Semprotkan pupuk organik cair (POC) atau asam amino secara berkala untuk menjaga daya tahan tanaman.",
            "Berikan mikroorganisme bermanfaat seperti Trichoderma dan mikoriza pada perakaran sebulan sekali."
        ],
        "prevention": [
            "Lakukan pemantauan rutin kebun minimal 2-3 kali seminggu untuk mendeteksi dini hama/penyakit.",
            "Pertahankan sanitasi bedengan dari gulma dan daun rontok.",
            "Jaga keteraturan penyiraman terutama saat memasuki fase berbunga dan pembentukan buah."
        ]
    },
    "Nutrition Deficiency": {
        "indonesian_name": "Defisiensi Nutrisi (Kekurangan Hara)",
        "scientific_name": "Nutritional Disorder / Abiotic Stress",
        "category": "Fisiologis / Hara",
        "severity": "Sedang",
        "severity_color": "#D97706",  # Warm Orange
        "icon_name": "flask",
        "summary": "Gangguan fisiologis akibat ketidakcukupan atau ketidakseimbangan unsur hara makro (N, P, K, Ca, Mg, S) atau mikro (Fe, Zn, B) dalam tanah atau akibat pH tanah ekstrem.",
        "symptoms": [
            "Defisiensi Nitrogen (N): Daun bagian bawah (daun tua) menguning merata dari ujung ke pangkal.",
            "Defisiensi Magnesium (Mg): Klorosis interveinal (tulang daun tetap hijau gelap, area antara tulang menguning).",
            "Defisiensi Kalium (K): Tepi dan ujung daun menguning lalu mengering cokelat seperti terbakar (marginal scorch).",
            "Defisiensi Kalsium (Ca): Daun muda melengkung kerdil, pucuk mati, dan memicu busuk pantat buah (blossom end rot).",
            "Defisiensi Besi (Fe): Daun pucuk muda memutih pucat dengan urat daun hijau halus."
        ],
        "cause": "Kandungan hara tanah minim, pH tanah terlalu asam (<5.5) atau terlalu basa (>7.5) yang mengikat kation hara, atau sistem perakaran rusak akibat kelebihan air/kekeringan.",
        "chemical_control": [
            "Semprotkan pupuk daun foliar makro & mikro lengkap dengan penyerapan cepat (misal Gandasil D/B, GrowMore).",
            "Untuk defisiensi Mg: Kocor atau semprot Magnesium Sulfat (Kieserite/Garam Epsom 2-3 g/L).",
            "Untuk defisiensi Ca: Aplikasikan Kalsium Nitrat (CNG) atau Kalsium Karbonat terlarut.",
            "Koreksi pH tanah dengan pengapuran dolomit bila tanah terlalu masam."
        ],
        "organic_control": [
            "Tambahkan kompos matang atau pupuk kandang terfermentasi yang kaya bahan organik (20-30 ton/ha).",
            "Kocorkan pupuk hayati cair berbasis bakteri pelarut fosfat dan penambat nitrogen.",
            "Gunakan ekstrak abu kayu kelapa untuk menambah unsur Kalium alami."
        ],
        "prevention": [
            "Lakukan uji tanah secara berkala sebelum memulai masa tanam.",
            "Pertahankan pH tanah di kisaran ideal 6.0 – 6.8.",
            "Terapkan pemupukan berimbang berbasis fase tanaman (N tinggi di fase awal, P & K dominan di fase berbuah).",
            "Perbaiki drainase tanah agar akar tidak tergenang dan membusuk."
        ]
    },
    "White spot": {
        "indonesian_name": "Bercak Putih",
        "scientific_name": "White Spot Necrosis / Fungal Blight",
        "category": "Jamur / Fisiologis",
        "severity": "Sedang",
        "severity_color": "#D97706",  # Amber
        "icon_name": "target",
        "summary": "Penyakit bercak putih pada daun cabai ditandai dengan lesi nekrotik keputihan yang tipis menyerupai kertas minyak, dapat dipicu oleh jamur patogen atau efek fitotoksisitas cairan pestisida.",
        "symptoms": [
            "Bercak-bercak putih pucat atau keperakan berdiameter kecil hingga sedang pada helai daun.",
            "Pusat bercak tampak kering, rapuh, dan tipis seperti selaput kertas transparan.",
            "Pada beberapa kasus, bagian tengah bercak putih rontok menyisakan lubang tembus (shot-hole effect).",
            "Bercak dapat membesar dan bergabung memicu kekeringan sebagian helai daun cabai."
        ],
        "cause": "Infeksi spora jamur bercak daun pada cuaca fluktuatif atau nekrosis sel akibat luka sengatan sinar matahari (sunscald) dan fitotoksisitas penyemprotan bahan kimia pekat.",
        "chemical_control": [
            "Semprotkan fungisida protektif berbahan aktif Propineb (Antracol 70WP) atau Mankozeb.",
            "Bila meluas, kombinasikan dengan fungisida sistemik golongan Triazol seperti Tebukonazol atau Difenokonazol.",
            "Pastikan dosis campuran pestisida tidak melebihi rekomendasi agar tidak memicu fitotoksik bercak putih."
        ],
        "organic_control": [
            "Aplikasikan semprotan larutan suspensi baking soda (Natrium Bikarbonat 3 g/L) + sabun hortikultura.",
            "Semprotkan biourin sapi terfermentasi yang telah diencerkan 1:10 sebagai pupuk dan bioprotektor.",
            "Aplikasi mikroba antagonis perakaran Trichoderma sp."
        ],
        "prevention": [
            "Hindari penyemprotan pestisida atau pupuk daun saat matahari terik di siang hari (semprot pagi sebelum jam 09.00 atau sore setelah jam 15.30).",
            "Gunakan nozel semprot berbutir halus (kabut) agar cairan tidak menumpuk membentuk titik pembakar di daun.",
            "Jaga jarak tanam dan pangkas daun tua yang terlalu rapat.",
            "Kumpulkan dan kubur daun yang terinfeksi bercak putih."
        ]
    }
}

def get_disease_details(class_name: str) -> dict:
    """Mengembalikan informasi penyakit berdasarkan nama kelas prediksi."""
    normalized = class_name.strip()
    if normalized in DISEASE_DATABASE:
        return DISEASE_DATABASE[normalized]
    
    for k, v in DISEASE_DATABASE.items():
        if k.lower() == normalized.lower():
            return v
            
    return {
        "indonesian_name": class_name,
        "scientific_name": "Tidak teridentifikasi spesifik",
        "category": "Informasi Umum",
        "severity": "Sedang",
        "severity_color": "#6B7280",
        "icon_name": "leaf",
        "summary": f"Hasil klasifikasi menunjukkan indikasi {class_name}.",
        "symptoms": ["Perhatikan perubahan warna, bentuk, atau tekstur pada helai daun."],
        "cause": "Dapat dipicu oleh patogen biotik atau stres lingkungan abiotik.",
        "chemical_control": ["Konsultasikan dengan petugas penyuluh pertanian setempat."],
        "organic_control": ["Jaga kebersihan lingkungan tanam dan gunakan pupuk kompos matang."],
        "prevention": ["Gunakan benih unggul, sanitasi lahan, dan perhatikan drainase tanah."]
    }
