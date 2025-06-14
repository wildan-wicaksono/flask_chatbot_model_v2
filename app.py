# Import libraries
import json, re, pickle, io, random
import numpy as np
from flask import Flask, request, jsonify

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# === Inisialisasi Flask App ===
app = Flask(__name__)

tips_responses = {
  "stress_due_to_academic": [
    "Tentu, terima kasih banyak atas kepercayaanmu, {user_name} 🙌 Aku memahami bahwa tekanan akademik bisa benar-benar sangat melelahkan—tidak hanya secara fisik, tetapi juga secara mental. Namun, ketahuilah bahwa kamu tidak sendirian dalam menghadapinya. Mari kita coba beberapa langkah kecil berikut ini untuk membantumu keluar dari tekanan tersebut secara perlahan:\n\n1. Pecah tugas-tugas besar menjadi bagian-bagian yang lebih kecil dan lebih mudah dikelola. Terkadang kita merasa kewalahan bukan karena tugasnya yang terlalu berat, melainkan karena kita melihatnya sebagai satu blok besar yang menakutkan. Cobalah untuk memecahnya menjadi bagian-bagian kecil dan fokus untuk mengerjakan satu per satu. Sedikit demi sedikit, lama-kelamaan pasti akan selesai juga.\n\n2. Prioritaskan tugas berdasarkan tingkat urgensi dan pentingnya. Kamu bisa mencoba menggunakan metode seperti Matriks Eisenhower, atau jika ingin cara yang lebih simpel, cukup tulis daftar tugas (to-do list) dan tandai mana yang paling mendesak untuk diselesaikan. Melihat progresmu dalam checklist tersebut juga bisa memberikan rasa kepuasan tersendiri, lho.\n\n3. Cobalah menerapkan teknik Pomodoro. Fokus untuk belajar atau mengerjakan tugas selama 25 menit, kemudian beristirahat selama 5 menit. Ulangi siklus ini sebanyak empat kali, lalu ambillah istirahat yang lebih panjang. Teknik ini dapat membantumu untuk tetap fokus tanpa merasa cepat kehabisan energi.\n\n4. Berikan penghargaan untuk dirimu sendiri atas setiap usaha yang telah dilakukan. Sekecil apa pun progres yang kamu capai, hal itu sangat layak untuk diapresiasi. Penghargaan ini bisa berupa menonton satu episode drama favorit, menikmati camilan kesukaan, atau sekadar beristirahat sambil rebahan. Memberikan reward juga merupakan salah satu bentuk self-love yang penting.\n\n5. Jangan pernah merasa harus menjalani semuanya sendirian. Terkadang, bercerita kepada teman, mentor, atau siapa pun yang kamu percaya bisa sangat membantu meringankan beban pikiranmu. Mendengarkan pendapat dari orang lain atau sekadar mendapatkan validasi atas perasaanmu adalah hal yang sangat penting.\n\nIngatlah selalu, kamu bukanlah seorang yang gagal jika membutuhkan waktu untuk istirahat. Kamu sedang berada dalam proses bertumbuh, dan proses itu tidak selalu berjalan mulus. Tetapi, kamu tidak sendirian dalam menjalaninya 🌱",
    "Terima kasih sudah mau membuka diri dan bercerita, {user_name}. Aku tahu kamu sudah berusaha sangat keras dan kamu sangat layak untuk mendapatkan apresiasi atas semua usahamu 💪✨ Mari kita coba beberapa strategi kecil berikut ini untuk membantumu tetap waras dan seimbang di tengah padatnya tuntutan dunia akademik:\n\n1. Atur target harian yang kecil dan realistis. Misalnya: 'hari ini aku cukup membaca 2 halaman materi', atau 'menyelesaikan 1 soal latihan saja sudah cukup'. Target-target kecil seperti ini akan membuatmu tetap berjalan maju tanpa merasa terbebani atau gagal.\n\n2. Gunakan pengatur waktu (timer) sebagai teman belajarmu. Teknik Pomodoro bisa menjadi penyelamat yang efektif. Belajarlah selama 25 menit, kemudian istirahat selama 5 menit. Metode ini membantu otakmu untuk tetap fokus dan tidak cepat merasa jenuh atau burnout.\n\n3. Temukan tempat belajar yang paling cocok dan nyaman untukmu. Ada orang yang lebih fokus belajar di kafe yang ramai, sementara yang lain membutuhkan suasana yang benar-benar sepi. Temukanlah 'ruang aman' untuk belajarmu tersebut.\n\n4. Berikan afirmasi positif kepada dirimu sendiri secara rutin. Cobalah untuk mengucapkan kalimat-kalimat ini secara perlahan: 'Aku boleh belajar dengan ritmeku sendiri dan itu tidak apa-apa', 'Aku sudah cukup baik', 'Aku sedang berproses dan berkembang'. Afirmasi seperti ini sangat membantu dalam menjaga kesehatan mentalmu.\n\n5. Rayakan setiap progres yang kamu capai, sekecil apa pun itu. Sudah berhasil membuka laptop untuk belajar? Itu adalah sebuah progres. Sudah membaca satu paragraf materi? Kamu hebat! Jangan menunggu sampai semuanya sempurna baru kamu mau mengapresiasi dirimu sendiri 🤍\n\nIngatlah, kamu bukanlah sekadar tugas-tugasmu. Kamu adalah seorang manusia yang sedang belajar dan bertumbuh."
  ],
  "stress_due_to_work": [
    "Tentu, terima kasih atas kepercayaanmu, {user_name}. Terkadang yang kita perlukan bukanlah perubahan besar yang drastis, melainkan langkah-langkah kecil yang dilakukan secara konsisten. Berikut adalah beberapa hal yang bisa kamu mulai terapkan untuk mengelola stres akibat pekerjaan:\n\n1. Buatlah daftar prioritas harian yang jelas. Cukup tuliskan tiga hal terpenting yang benar-benar ingin kamu selesaikan hari ini. Hindari membuat daftar yang terlalu panjang agar kamu tidak merasa kewalahan sejak awal.\n\n2. Manfaatkan teknik Pomodoro untuk menjaga fokus dan energi. Aturlah timer selama 25 menit untuk bekerja dengan konsentrasi penuh, kemudian berikan dirimu waktu istirahat selama 5 menit. Lakukan siklus ini sebanyak empat kali, lalu ambillah istirahat yang lebih panjang. Metode ini terbukti membantu menjaga fokus dan mencegah kelelahan.\n\n3. Ciptakan sebuah rutinitas 'penutup kerja' yang menandakan berakhirnya jam kerjamu. Setelah selesai bekerja, lakukan satu hal yang secara simbolis menandakan bahwa kamu 'selesai' untuk hari itu, seperti mandi air hangat, menyeduh teh herbal, atau mematikan laptop kerjamu. Hal ini membantu otakmu untuk beralih ke mode istirahat.\n\n4. Luangkan waktu sekitar 10 menit untuk menuliskan isi pikiranmu dalam sebuah jurnal. Biarkan semua unek-unek, daftar tugas yang belum selesai, atau ketakutan yang mungkin kamu rasakan tertuang bebas. Ini adalah cara yang sangat baik untuk merapikan pikiran dan melepaskan beban.\n\n5. Validasi perasaanmu dan berikan dirimu izin untuk merasa lelah. Tidak ada keharusan untuk selalu produktif setiap saat, dan sangat wajar jika kamu merasa capek. Terkadang kita hanya perlu diingatkan bahwa istirahat juga merupakan bagian penting dari sebuah progres dan produktivitas jangka panjang 🌿",
    "Tentu, {user_name}, aku senang kamu bersedia mencoba. Stres di tempat kerja itu hal yang umum, tapi bukan berarti tidak bisa dikelola. Mari kita coba beberapa strategi kecil ini yang mungkin bisa membantu meringankan bebanmu:\n\n1. Fokus pada satu tugas dalam satu waktu. Hindari multitasking karena seringkali malah membuat pekerjaan terasa lebih berat dan hasilnya kurang maksimal. Selesaikan satu hal dulu baru beralih ke yang lain.\n\n2. Tetapkan batasan yang jelas antara pekerjaan dan kehidupan pribadi. Misalnya, tentukan jam berapa kamu benar-benar berhenti bekerja dan matikan notifikasi pekerjaan setelah jam tersebut. Ini penting untuk mencegah burnout.\n\n3. Luangkan waktu untuk melakukan aktivitas fisik ringan secara teratur. Sekadar berjalan kaki singkat di sela-sela pekerjaan atau peregangan ringan bisa membantu melepaskan ketegangan dan menjernihkan pikiran.\n\n4. Pastikan kamu mendapatkan istirahat yang cukup dan berkualitas. Kurang tidur dapat memperburuk tingkat stres dan menurunkan produktivitas. Usahakan untuk memiliki jadwal tidur yang teratur.\n\n5. Jangan ragu untuk berbicara dengan atasan atau rekan kerja jika kamu merasa beban kerjamu terlalu berat. Terkadang, komunikasi yang terbuka bisa menghasilkan solusi yang tidak terpikirkan sebelumnya. Ingatlah, meminta bantuan bukanlah tanda kelemahan, melainkan langkah bijak untuk menjaga kesejahteraan dirimu 💪✨"
  ],
  "stress_due_to_family": [
    "Kamu sungguh kuat telah berani menceritakan hal ini, terima kasih banyak atas kepercayaanmu 🙏 Terkadang, hubungan dalam keluarga bisa menjadi sangat rumit dan benar-benar menguras emosi. Berikut adalah beberapa hal yang bisa kamu coba terapkan secara perlahan untuk membantu dirimu sendiri:\n\n1. Jika kamu merasa cukup aman dan siap, cobalah untuk menerapkan komunikasi yang asertif — ungkapkan perasaanmu dengan jujur tanpa menyalahkan pihak lain. Mulailah kalimatmu dengan 'aku merasa...', bukan dengan 'kamu selalu...'.\n2. Tuliskan semua unek-unek atau beban pikiranmu dalam sebuah jurnal. Jangan disaring atau diedit terlebih dahulu, biarkan semuanya keluar apa adanya. Cara ini dapat membantumu untuk mengenali pola-pola tertentu dan meringankan beban pikiran yang ada.\n3. Bangun batasan-batasan yang sehat (boundaries) dalam hubunganmu. Kamu berhak untuk memiliki ruang emosional pribadi, bahkan dari orang-orang terdekat sekalipun. Ini penting untuk menjaga kewarasanmu.\n4. Sisihkan waktu khusus untuk dirimu sendiri, walaupun hanya 10 menit di dalam kamar, di kamar mandi, atau sekadar berjalan-jalan sebentar di luar rumah. Mengisi ulang energi (recharge) bukanlah bentuk egoisme — melainkan sebuah bentuk penyelamatan diri yang sangat penting.\n5. Apabila semuanya terasa terlalu berat untuk dihadapi sendirian, ingatlah bahwa mencari bantuan profesional bukanlah berarti kamu lemah. Justru, itu adalah pertanda bahwa kamu menyayangi dirimu sendiri dan ingin menjadi lebih baik 🌱",
    "Terima kasih sudah mau berbagi cerita denganku, {user_name}. Permasalahan dalam keluarga itu memang tidak pernah mudah untuk dihadapi, apalagi jika kamu merasa sendirian di dalamnya. Jika kamu berkenan, berikut adalah beberapa cara yang bisa kamu coba untuk membantumu melewati ini:\n\n1. Tuliskan sebuah surat (boleh saja tidak kamu kirimkan) yang ditujukan kepada orang yang membuatmu merasa lelah atau tertekan. Tuliskan semua perasaanmu dengan sejujurnya, agar emosimu memiliki tempat untuk keluar dan tidak terpendam.\n2. Latihlah kemampuan untuk mengatakan 'tidak' dengan cara yang lembut namun tetap tegas — misalnya, 'Aku mengerti maksudmu, tetapi aku belum siap untuk membicarakan hal itu sekarang.' atau 'Terima kasih atas sarannya, namun aku perlu waktu untuk memikirkannya sendiri.'\n3. Bangun sebuah rutinitas mini yang sepenuhnya hanya untuk dirimu sendiri — seperti waktu mandi yang tenang dan menenangkan, kegiatan journaling untuk merefleksikan perasaan, atau mendengarkan musik favoritmu. Hal-hal kecil ini bisa menjadi semacam jangkar emosional ketika suasana rumah terasa begitu chaos.\n4. Identifikasi pola-pola konflik yang sering terjadi: kapan biasanya konflik itu mulai muncul? Dengan siapa konflik tersebut sering terjadi? Setelah kejadian apa konflik itu biasanya terpicu? Memahami hal ini dapat membantumu untuk mengerti ritme dinamika keluarga dan belajar untuk mengantisipasinya.\n5. Validasi perasaan ini dalam dirimu: kamu berhak untuk merasa aman dan nyaman — meskipun orang lain mungkin berkata 'itu kan hanya keluargamu sendiri'. Kamu tetaplah seorang individu yang membutuhkan perlindungan, rasa hormat, dan penghargaan 💛"
  ],
  "stress_due_to_relationship": [
    "Tentu, terima kasih sudah mau terbuka dan mempercayaiku, {user_name} 🤍 Terkadang sebuah hubungan bisa sangat menguras energi, bukan karena kita yang lemah, melainkan karena kita mungkin terlalu banyak memberi tanpa mendapatkan kembali sepadan. Berikut adalah beberapa hal yang bisa membantumu untuk menjaga diri sendiri terlebih dahulu dalam situasi ini:\n\n1. Cobalah untuk menuliskan batasan-batasan pribadimu secara jelas. Hal apa saja yang membuatmu merasa tidak nyaman dalam hubungan ini? Mulailah dari hal-hal kecil, misalnya dengan menetapkan bahwa 'aku membutuhkan waktu untuk diriku sendiri setelah bertengkar agar bisa menenangkan pikiran'.\n2. Lakukan evaluasi ulang terhadap perasaanmu setelah setiap interaksi dengan pasangan. Apabila kamu secara konsisten merasa lelah, takut, atau tidak dihargai setelahnya, itu bisa menjadi sinyal penting yang perlu kamu perhatikan.\n3. Jagalah koneksi sosialmu yang lain di luar hubungan ini. Kehadiran teman-teman dan keluarga bisa menjadi semacam jangkar atau penyeimbang realitas ketika hubunganmu mulai terasa menenggelamkan atau tidak sehat.\n4. Jangan pernah mengabaikan sinyal-sinyal yang diberikan oleh tubuhmu. Sakit kepala yang sering muncul, kesulitan tidur di malam hari, atau bahkan rasa mual yang tidak jelas penyebabnya bisa jadi merupakan reaksi tubuh terhadap tekanan emosional yang sedang kamu alami.\n5. Ingatlah bahwa kamu sama sekali tidak egois karena memilih untuk menjaga jarak atau menetapkan batasan. Itu justru merupakan pertanda bahwa kamu mulai belajar untuk lebih mencintai dan menghargai dirimu sendiri 🌿",
    "Terima kasih banyak telah percaya untuk meminta bantuanku, {user_name}. Mari kita coba beberapa langkah kecil berikut ini yang mungkin bisa membantumu untuk tetap merasa utuh dan menjaga kewarasan di tengah dinamika hubungan yang mungkin sedang membuatmu merasa lelah:\n\n1. Cobalah untuk menyadari pola-pola yang berulang dalam hubunganmu. Apakah kamu selalu merasa menjadi pihak yang salah? Atau apakah kamu harus selalu meminta maaf terlebih dahulu meskipun itu bukanlah kesalahanmu? Mengenali pola ini adalah langkah awal.\n2. Buatlah sebuah jurnal khusus mengenai relasimu. Tuliskan setiap kali kamu merasa tidak aman, tidak nyaman, atau tidak dihargai dalam hubungan tersebut. Catatan ini bisa membantumu melihat gambaran yang lebih jelas.\n3. Beranikan dirimu untuk mengatakan ‘tidak’ pada hal-hal yang memang tidak sesuai dengan nilai atau keinginanmu. Ingatlah bahwa batasan bukanlah sebuah dinding yang memisahkan, melainkan sebuah pagar yang melindungi agar kamu tetap merasa aman dan nyaman.\n4. Berikan ruang untuk dirimu mengambil jeda sejenak. Kamu sangat boleh untuk berhenti membalas pesan atau mengangkat telepon untuk sementara waktu demi menjaga kewarasan dan ketenangan batinmu.\n5. Rayakan setiap kemajuan yang berhasil kamu capai. Sekecil apa pun langkah yang kamu ambil untuk menjauh dari pola hubungan yang toksik atau tidak sehat, itu adalah sebuah bentuk keberanian yang luar biasa 💪"
  ],
  "stress_due_to_life_pressure": [
    "Tekanan hidup itu terkadang bisa terasa seperti kabut tebal yang menghalangi pandangan, ya? Tidak berbentuk jelas, namun dampaknya begitu nyata, membuat napas terasa lebih berat dan setiap langkah terasa sulit. Jika kamu merasakan hal serupa, mari kita coba beberapa pendekatan yang mungkin bisa membantu meringankan beban tersebut sedikit demi sedikit:\n\n1. Cobalah untuk membuat sebuah daftar sederhana mengenai hal-hal yang telah berhasil kamu kerjakan atau lalui selama seminggu terakhir, sekecil apa pun pencapaian itu. Dengan melihat daftar tersebut, kamu mungkin akan menyadari betapa banyaknya rintangan yang sudah kamu taklukkan, meskipun dengan perasaan lelah yang menyertai.\n2. Penting untuk diingat bahwa laju kehidupan setiap individu itu unik dan berbeda. Hidup ini bukanlah sebuah arena perlombaan, melainkan sebuah perjalanan pribadi yang penuh dengan kesempatan untuk beristirahat dan memetik pelajaran berharga di setiap pemberhentian.\n3. Ciptakanlah rutinitas-rutinitas kecil yang dapat memberikan rasa stabilitas dan ketenangan dalam keseharianmu. Aktivitas sederhana seperti menikmati mandi air hangat di penghujung hari, merapikan tempat tidur setiap pagi, atau meluangkan waktu untuk menikmati secangkir teh hangat di sore hari, bisa menjadi jangkar emosional di tengah hari-hari yang seringkali terasa tidak pasti.\n4. Coba identifikasi satu saja beban pikiran atau tanggung jawab yang rasanya bisa kamu lepaskan atau kurangi intensitasnya hari ini. Mungkin ada sesuatu yang tidak harus kamu pikul sepenuhnya sendirian, atau ada tugas yang sebenarnya bisa ditunda tanpa konsekuensi besar.\n5. Dan yang paling krusial untuk diingat, {user_name}, kamu bukanlah sebuah kegagalan hanya karena perjalananmu terasa berbeda dari orang lain atau karena kamu belum mencapai semua tujuan yang kamu inginkan. Kamu sedang berada dalam sebuah proses pembelajaran untuk bertahan dan bertumbuh di dunia yang terkadang menuntut begitu banyak. 🌾",
    "Ada kalanya dunia ini terasa berputar terlalu cepat, dan kita merasa kewalahan hanya untuk sekadar bisa tetap berdiri tegak. Jika itu yang sedang kamu rasakan, cobalah beberapa langkah kecil ini untuk membantumu menemukan pijakan:\n\n1. Luangkan waktu sejenak untuk bertanya kepada dirimu sendiri: 'Siapakah aku sebenarnya, di luar semua pencapaian atau tuntutan produktivitas?' Jawaban yang muncul dari dalam dirimu itu sangatlah penting.\n2. Ketika pikiranmu mulai terasa penuh dan berputar tanpa henti (overthinking), berhentilah sejenak. Letakkan tanganmu di dada, rasakan detak jantungmu, dan ucapkan dengan lembut kepada diri sendiri: 'Aku ada di sini, saat ini, dan aku sudah cukup.'\n3. Ekspresikan isi kepalamu tanpa aturan atau sensor. Kamu bisa menggambarnya, menuliskannya dalam bentuk coretan bebas, atau apa pun yang membuatmu merasa lega. Biarkan semua sesak itu menemukan jalannya untuk keluar.\n4. Coba ganti 'daftar pekerjaan' (to-do list) dengan 'daftar keinginan menjadi' (to-be list). Misalnya, hari ini kamu bisa menulis: ‘Hari ini aku ingin menjadi seseorang yang lebih sabar dan berbelas kasih kepada diriku sendiri.’\n5. Berlatihlah untuk menerima bahwa akan selalu ada hari-hari yang terasa buruk atau tidak sesuai harapan. Mengalami hari buruk tidak menjadikanmu gagal; itu justru membuatmu semakin nyata sebagai seorang manusia. 🌙"
  ],
  "anxiety_due_to_expectation": [
    "Tentu, {user_name}. Sangat wajar merasa terbebani ketika ekspektasi dari orang lain terasa begitu tinggi. Kadang ekspektasi tersebut bisa membuat kita lupa akan nilai dan keinginan diri sendiri. Cobalah untuk membuat batasan-batasan kecil terlebih dahulu — bedakan mana yang benar-benar menjadi keinginanmu, dan mana yang mungkin lebih merupakan harapan orang lain. Ingatlah bahwa setiap langkah kecil yang kamu ambil untuk dirimu sendiri itu sangatlah valid dan berarti.",
    "Kamu berhak untuk merasa gagal, lelah, atau bahkan ingin berhenti sejenak, dan itu tidak akan mengurangi nilaimu sebagai manusia, {user_name}. Tidak apa-apa, semua perasaan itu sangatlah manusiawi. Cobalah untuk tidak memaksakan dirimu untuk selalu sempurna atau memenuhi semua standar yang ada. Kamu sudah cukup berharga apa adanya. 🌱"
  ],
  "anxiety_due_to_social": [
    "Tentu, terima kasih banyak sudah mau terbuka dan berbagi, {user_name}. Aku sungguh memahami bahwa rasa cemas ketika berada dalam situasi sosial atau harus berinteraksi dengan orang lain bisa terasa luar biasa berat dan melelahkan. Namun, ketahuilah bahwa secara perlahan, kamu pasti bisa mengatur napasmu kembali dan merasa lebih tenang. Berikut adalah beberapa tips yang lebih mendalam dan mungkin bisa kamu coba:\n\n1.  Cobalah untuk mengenali pemicu (trigger) kecemasanmu. Kapan biasanya kamu mulai merasa cemas? Apakah saat harus memulai sebuah obrolan? Atau ketika merasa menjadi pusat perhatian dan mendapatkan tatapan dari banyak orang? Dengan mencatat pola-pola ini, kamu bisa lebih memahami dinamika kecemasanmu dan mempersiapkan diri lebih baik.\n2.  Latihlah dirimu dengan melakukan paparan (exposure) kecil dan bertahap terhadap situasi sosial setiap harinya. Misalnya, mulailah dengan menyapa penjaga toko dengan ramah, atau memberanikan diri untuk bertanya arah kepada petugas di suatu tempat. Mulailah dari interaksi-interaksi yang singkat, terasa relatif aman, dan memiliki risiko rendah bagimu.\n3.  Persiapkan sebuah ‘safety script’ atau beberapa kalimat aman yang bisa kamu gunakan sebelum kamu terlibat dalam sebuah interaksi sosial yang lebih kompleks. Contohnya: 'Hai, bolehkah aku bertanya sesuatu padamu?', atau 'Apakah kamu sering berkunjung ke tempat ini ya? Aku baru pertama kali.' Skrip seperti ini dapat membantu mengurangi beban pikiranmu untuk harus selalu berimprovisasi secara spontan saat berbicara.\n4.  Manfaatkan teknik grounding 5-4-3-2-1 untuk menenangkan diri ketika rasa cemas mulai meningkat. Sebutkan dalam hati atau dengan suara pelan: 5 hal yang kamu lihat di sekitarmu, 4 benda yang bisa kamu rasakan sentuhannya, 3 suara yang kamu dengar, 2 aroma yang bisa kamu cium, dan 1 hal yang bisa kamu rasakan di mulutmu (misalnya rasa permen atau air liur).\n5.  Cobalah untuk membingkai ulang pola pikirmu (reframe your mindset) mengenai interaksi sosial. Ingatlah bahwa jika sebuah interaksi tidak berjalan selancar yang kamu harapkan, itu bukan berarti kamu adalah pribadi yang aneh atau gagal — itu hanyalah pertanda bahwa kamu sedang dalam proses belajar untuk membentuk relasi yang lebih baik dan lebih nyaman. Setiap pengalaman adalah guru. 🌱",
    "Aku sangat mengerti, {user_name}, bahwa interaksi sosial itu terkadang bisa terasa seperti sebuah tantangan berat yang harus dihadapi setiap harinya, dan kecemasan yang menyertainya bisa sangat menguras energi. Tetapi, kamu sudah menunjukkan hal yang luar biasa karena mau belajar untuk menghadapinya dan mencari cara untuk merasa lebih baik. Berikut adalah beberapa langkah yang mungkin bisa membantumu untuk merasa lebih percaya diri dan tenang:\n\n1.  Buatlah sebuah jurnal khusus mengenai kecemasan sosialmu. Di dalamnya, kamu bisa menuliskan kapan kamu merasa cemas, apa saja yang kamu rasakan secara fisik (misalnya jantung berdebar, tangan berkeringat) maupun emosional (misalnya takut, malu), dan bagaimana akhirnya situasi tersebut berjalan. Jurnal ini dapat membantumu untuk melihat progres dirimu dari waktu ke waktu dan mengenali pola-pola tertentu.\n2.  Lakukan visualisasi positif di mana dirimu berhasil melewati situasi sosial tersebut dengan sukses, tenang, dan percaya diri. Setiap malam sebelum tidur, atau sebelum menghadapi situasi sosial tertentu, bayangkan dirimu bisa tampil dengan tenang, berbicara dengan lancar, dan bahkan merasa dihargai serta diterima oleh orang lain. Latihan mental ini bisa membantu membangun kesiapanmu.\n3.  Temukan setidaknya satu tempat atau lingkungan yang terasa aman dan suportif untukmu. Ini bisa berupa seorang teman dekat yang bisa memahami dan mendukungmu, sebuah kelas atau komunitas yang sesuai dengan minatmu (di mana kamu bisa merasa lebih nyaman karena kesamaan minat), atau bahkan sebuah forum online yang suportif. Latihlah kemampuan interaksimu di sana terlebih dahulu sebelum mencoba di lingkungan yang lebih menantang.\n4.  Gantilah percakapan negatif dengan diri sendiri (negative self-talk) yang sering muncul menjadi kalimat-kalimat yang lebih positif, realistis, dan membangun. Misalnya, dari pikiran seperti 'Aku pasti akan terlihat aneh dan canggung saat berbicara nanti' menjadi 'Aku mungkin akan merasa sedikit gugup, tetapi itu adalah hal yang sangat manusiawi dan wajar. Aku akan berusaha yang terbaik.'\n5.  Berikan penghargaan (reward) yang tulus kepada dirimu sendiri setiap kali kamu berhasil menghadapi atau mengatasi rasa cemasmu dalam situasi sosial, sekecil apapun pencapaiannya. Penghargaan ini tidak perlu selalu berupa hal yang besar — bisa saja secangkir kopi favoritmu, waktu istirahat sejenak tanpa ada rasa bersalah sedikit pun, atau sekadar pujian tulus untuk dirimu sendiri. ☕"
  ],
  "anxiety_due_to_future": [
    "Aku sungguh memahami usahamu untuk tetap menjaga kewarasan di tengah gelombang ketidakpastian ini, {user_name}. Masa depan memang seringkali terasa seperti melangkah di jalan yang gelap, di mana semuanya serba tidak pasti dan bayangan kekhawatiran mudah muncul. Tapi, kamu tidak harus melaluinya sendirian, dan kamu selalu bisa memulai dari langkah-langkah terkecil untuk menemukan sedikit cahaya. Berikut beberapa hal yang bisa kamu coba lakukan:\n\n1.  Cobalah menulis sebuah surat dari 'versi dirimu di masa depan yang lebih bijak dan tenang' untuk dirimu yang sekarang. Bayangkan dia berbisik padamu: 'Kamu tidak harus mengetahui semua jawaban sekarang. Setiap langkah kecil yang kamu ambil dengan penuh kesadaran hari ini adalah fondasi yang membuatku bisa bertahan dan bahkan berkembang.'\n2.  Luangkan waktu untuk membuat tiga skenario masa depan: satu yang paling realistis berdasarkan kondisimu saat ini, satu yang ideal sesuai impian terdalammu, dan satu lagi yang mungkin terasa sedikit 'gila' atau di luar kebiasaan. Latihan ini bisa membantumu menyadari bahwa masa depan bukanlah satu titik akhir yang kaku, melainkan sebuah spektrum kemungkinan yang luas dan bisa dibentuk.\n3.  Tetapkan tujuan mingguan dengan tema tertentu yang relevan dengan apa yang ingin kamu eksplorasi atau perbaiki. Contohnya, minggu ini kamu bisa fokus pada 'eksplorasi minat baru'. Minggu depannya, mungkin 'melatih konsistensi dalam satu kebiasaan kecil'. Tema ini memberi arah tanpa terlalu membebani.\n4.  Ambil satu langkah eksplorasi konkret setiap minggu: mungkin dengan mengikuti webinar gratis, memberanikan diri mengobrol dengan orang baru yang inspiratif, atau mencoba aktivitas yang belum pernah kamu lakukan sebelumnya. Tujuan utamanya bukan untuk langsung menemukan jawaban, tetapi untuk memperluas horizon pemikiran dan pengalamanmu.\n5.  Bangun sebuah 'sistem refleksi' singkat setiap malam sebelum tidur. Cukup luangkan 5-10 menit untuk merenungkan: Apa pelajaran berharga yang aku dapatkan hari ini? Bagaimana perasaanku mengenai masa depan saat ini setelah melewati hari ini? Refleksi ini membantu menenangkan pikiran dan melihat progres kecil.\n\nTidak apa-apa jika jalan di depan masih terasa gelap. Setiap langkah yang kamu ambil, sekecil apapun, tetaplah sah dan berarti dalam perjalananmu. ✨",
    "Terima kasih sudah bersedia terbuka dan mencari cara untuk menghadapi kecemasan ini, {user_name}. Terkadang masa depan itu terasa seperti sebuah puzzle raksasa tanpa ada petunjuk yang jelas, dan sangat wajar jika kamu merasa bingung atau kewalahan. Tapi, kamu tidak harus menyelesaikan semua potongan puzzle itu sekaligus. Mari kita coba pecah menjadi bagian-bagian yang lebih kecil dan lebih mudah dihadapi:\n\n1.  Cobalah untuk menentukan satu hal yang paling kamu inginkan untuk dirasakan di masa depan—ini bukan tentang pencapaian materi atau status, melainkan tentang kondisi batin. Misalnya: perasaan damai, kebebasan berekspresi, atau rasa dihargai. Jadikan perasaan ini sebagai kompas batinmu.\n2.  Luangkan waktu untuk menulis ulang definisi ‘sukses’ atau ‘hidup yang bermakna’ versi kamu sendiri. Jangan terpaku pada template atau standar yang mungkin ditetapkan oleh orang lain atau lingkungan. Definisimu adalah yang paling penting.\n3.  Buatlah sebuah ‘daftar keingintahuan’ (curiosity list), bukan hanya sekadar daftar karier atau target formal. Misalnya, kamu mungkin tertarik untuk memahami bagaimana sistem kerja industri kreatif, atau mengapa orang sering merasa insecure, atau bagaimana cara berkebun di lahan sempit. Dari daftar ini, kamu bisa menemukan arah baru untuk belajar dan berkembang.\n4.  Mulailah membangun satu kebiasaan kecil yang positif, cukup 15 menit setiap hari. Konsistensi dalam hal kecil seringkali jauh lebih berdampak dan berkelanjutan daripada membuat resolusi besar yang kemudian menguap begitu saja.\n5.  Jika memungkinkan dan kamu merasa nyaman, jangan ragu untuk berkonsultasi dengan seorang mentor, dosen yang kamu percaya, atau bahkan seorang profesional di bidang pengembangan diri. Terkadang, pandangan dan insight dari mereka bisa membuka jalan atau perspektif baru yang tidak terpikirkan olehmu sebelumnya.\n\nIngatlah selalu, {user_name}, menemukan arah hidup itu bukanlah soal seberapa cepat kamu berlari, melainkan seberapa jujur kamu terhadap dirimu sendiri dalam setiap langkahnya. 🌱"
  ],
  "anxiety_due_to_failure": [
    "Terima kasih banyak sudah mau terbuka dan berbagi perasaanmu denganku, {user_name}. Rasa takut akan kegagalan itu adalah hal yang sangat manusiawi, apalagi ketika kamu begitu peduli dengan hasil dari apa yang sedang kamu kerjakan atau impikan. Namun, ingatlah bahwa kamu juga sangat berhak untuk merasa aman, tenang, dan berharga, bahkan di tengah berbagai ketidakpastian yang ada. Mari kita coba beberapa langkah berikut ini bersama-sama untuk membantumu mengelola perasaan tersebut:\n\n1.  Cobalah untuk mengubah bahasa internal yang kamu gunakan kepada dirimu sendiri. Daripada mengatakan 'aku gagal' atau 'aku pecundang', cobalah untuk menggantinya dengan kalimat yang lebih suportif dan berorientasi pada proses, seperti 'aku sedang dalam proses belajar dan bertumbuh dari pengalaman ini'. Ingatlah bahwa bahasa yang kita gunakan itu sangat membentuk cara kita berpikir, merasa, dan bertindak.\n2.  Luangkan waktu untuk menulis ulang kisah kegagalanmu dari sudut pandang pembelajaran yang berharga. Tanyakan pada dirimu: Hal apa saja yang aku pahami dengan lebih baik sekarang setelah melalui pengalaman tersebut? Aspek apa dari diriku yang justru menjadi lebih kuat atau lebih bijaksana karenanya? Pelajaran apa yang bisa aku bawa untuk langkah selanjutnya?\n3.  Buatlah sebuah daftar atau catatan kecil berisi momen-momen di masa lalu di mana kamu pernah mencoba melakukan sesuatu meskipun kamu merasa takut atau ragu. Itu adalah sebuah bentuk keberanian yang nyata, dan setiap keberanian, sekecil apapun, sangat layak untuk dihargai, diakui, dan dirayakan sebagai bagian dari kekuatanmu.\n4.  Latihlah dirimu untuk menghadapi 'kegagalan kecil' atau ketidaksempurnaan dalam aktivitas sehari-hari secara sadar dan terkontrol. Cobalah hal-hal baru tanpa menaruh ekspektasi yang terlalu tinggi terhadap hasilnya; fokuslah pada proses mencoba dan belajar. Ini bisa membantu pikiranmu untuk belajar bahwa kegagalan itu bukanlah sesuatu yang mengerikan atau akhir dari segalanya, melainkan bagian alami dari pertumbuhan.\n5.  Ingatkan dirimu sendiri secara terus-menerus bahwa nilaimu sebagai seorang individu tidaklah ditentukan semata-mata oleh hasil dari usahamu atau pencapaianmu. Kamu tetaplah pribadi yang pantas untuk dicintai, dihargai, dan diterima apa adanya, bahkan ketika hasil yang kamu dapatkan tidak selalu sesuai dengan harapanmu. 💛",
    "Kamu sudah berhasil sampai pada titik ini, {user_name}—itu artinya kamu adalah pribadi yang kuat, tangguh, dan memiliki keinginan untuk bertumbuh. Rasa takut akan kegagalan memang bisa membuat kita merasa stagnan dan tidak berani melangkah maju, tetapi kesadaranmu akan perasaan tersebut saja sudah merupakan sebuah langkah besar menuju perubahan yang positif. Mari kita coba beberapa tips berikut ini untuk membantumu dalam proses ini:\n\n1.  Cobalah untuk memisahkan dengan jelas antara proses yang kamu jalani dengan hasil akhir yang kamu dapatkan. Mengalami sebuah kegagalan bukan berarti kamu tidak berusaha dengan keras atau tidak memiliki kemampuan; seringkali itu berarti bahwa hasil tidak selalu sejalan dengan niat baik atau usaha maksimal yang telah kita curahkan, dan ada faktor-faktor lain yang mungkin berpengaruh.\n2.  Perhatikan suara-suara negatif atau kritikus internal yang seringkali muncul di dalam kepalamu ketika kamu memikirkan kegagalan. Cobalah untuk menantang balik suara-suara tersebut dengan logika, fakta, dan pertanyaan reflektif, misalnya dengan bertanya: 'Apakah benar semua orang di sekitarku menganggapku gagal total? Ataukah itu hanyalah pikiranku sendiri yang mungkin terlalu kritis dan perfeksionis?'\n3.  Ciptakan dan latih afirmasi harian yang positif, realistis, dan menguatkan untuk dirimu sendiri. Contohnya: 'Aku boleh saja melakukan kekeliruan atau kesalahan dalam perjalananku. Aku tetaplah seorang manusia yang berharga, layak untuk bahagia, dan mampu untuk belajar serta bangkit kembali.' Ucapkan dengan penuh keyakinan.\n4.  Buatlah target-target kecil yang lebih berfokus pada proses dan usaha, bukan hanya pada hasil akhir yang mungkin di luar kendalimu. Contohnya: 'Hari ini aku mau mencoba untuk melakukan satu kali lagi usaha ini dengan sebaik mungkin, tanpa terlalu merasa takut atau terbebani akan hasilnya nanti. Aku akan fokus pada apa yang bisa aku kontrol.'\n5.  Temukan sebuah ruang aman untuk dirimu—ini bisa berupa sebuah komunitas yang suportif, teman-teman yang bisa dipercaya dan tidak menghakimi, atau bahkan kegiatan journaling yang rutin. Memiliki tempat di mana kamu boleh jujur dengan perasaanmu tanpa merasa takut akan dihakimi adalah hal yang sangat penting untuk pemulihan. 🌱"
  ],
  "self_worth_low_confidence": [
    "Terima kasih sudah bersedia terbuka dan menunjukkan keinginan untuk bertumbuh, {user_name}. 💛 Membangun kembali rasa percaya diri dan harga diri adalah sebuah perjalanan, dan setiap langkah kecil sangatlah berarti. Berikut beberapa hal yang bisa kamu coba terapkan secara perlahan dalam keseharianmu:\n\n1.  Setiap hari, cobalah untuk menuliskan setidaknya 3 hal kecil yang telah kamu lakukan dengan baik atau yang membuatmu merasa sedikit bangga pada dirimu. Ini tidak harus hal besar; bahkan menyelesaikan tugas kecil seperti membereskan kamar, bangun pagi tepat waktu, atau berani membalas chat penting itu sudah merupakan pencapaian yang valid dan layak diapresiasi.\n2.  Latihlah dirimu untuk berhenti membandingkan progres atau perjalanan hidupmu dengan orang lain, terutama apa yang kamu lihat di media sosial. Ingatlah bahwa setiap orang memiliki garis waktu dan tantangannya masing-masing. Fokuslah untuk membandingkan dirimu hari ini dengan versi dirimu di hari kemarin; lihatlah pertumbuhanmu sendiri.\n3.  Ucapkan afirmasi yang lembut, positif, dan realistis kepada dirimu sendiri setiap pagi atau sebelum tidur. Mulailah dari kalimat-kalimat yang simpel namun kuat, seperti: 'Aku sudah cukup baik apa adanya', 'Aku sedang berproses dan belajar setiap hari', 'Aku layak untuk mendapatkan kasih sayang dan penghargaan, termasuk dari diriku sendiri'.\n4.  Ketika seseorang memberikan pujian atau apresiasi kepadamu, cobalah untuk menerimanya dengan tulus dan mengucapkan ‘terima kasih’. Kamu tidak harus langsung mempercayainya 100% jika masih terasa sulit, tetapi hindari untuk secara otomatis menolak atau meremehkan semua bentuk validasi positif yang datang dari luar.\n5.  Berikan dirimu sendiri izin untuk melakukan kesalahan atau mengalami kegagalan. Ingatlah bahwa kegagalan bukanlah bukti bahwa kamu adalah pribadi yang buruk atau tidak mampu, melainkan sebuah bukti bahwa kamu sudah berani untuk mencoba dan belajar. 🌱",
    "Ini adalah beberapa cara yang bisa membantumu untuk mulai membangun kembali rasa percaya diri dan menghargai dirimu sendiri, {user_name}. Ingatlah, lakukan satu langkah kecil dalam satu waktu, dan bersabarlah dengan prosesmu:\n\n1.  Praktikkan bahasa tubuh yang menunjukkan rasa percaya diri, meskipun mungkin di dalam hati kamu masih merasa ragu. Cobalah untuk berdiri atau duduk dengan tegak, lakukan kontak mata secukupnya saat berbicara, dan berikan senyuman tulus. Bahasa tubuh ini bisa mengirimkan sinyal positif ke otakmu bahwa kamu merasa aman dan berdaya.\n2.  Kurangi paparan terhadap akun-akun media sosial atau konten-konten yang seringkali membuatmu merasa kecil, minder, atau tidak cukup. Jaga 'asupan' informasi dan gambaran yang masuk ke dalam pikiranmu, sama seperti kamu menjaga asupan makanan untuk tubuhmu. Pilih konten yang membangun dan positif.\n3.  Tantang dirimu untuk melakukan satu ketakutan sosial kecil setiap minggunya. Misalnya, memberanikan diri untuk menjawab panggilan Zoom dengan kamera menyala, mengomentari postingan teman dengan suportif, atau sekadar memulai obrolan singkat selama 1 menit dengan orang baru di antrean kasir. Keberhasilan kecil ini akan membangun momentum.\n4.  Buatlah sebuah ‘daftar kemenangan kecil’ atau ‘jurnal pencapaian harian’. Catat setiap hal positif yang berhasil kamu lakukan atau lewati, sekecil apapun itu. Daftar ini akan membuatmu sadar bahwa kamu tidak seburuk atau segagal yang mungkin sering kamu pikirkan.\n5.  Ingatlah selalu prinsip ini: kamu tidak harus menunggu sampai dirimu menjadi sempurna terlebih dahulu untuk layak dihargai dan dicintai. Kamu sudah layak mendapatkan semua itu, bahkan ketika kamu merasa masih berantakan atau sedang dalam proses perbaikan diri. 🤍"
  ],
  "self_worth": [
    "Terima kasih sudah percaya dan bersedia untuk memulai perjalanan ini, {user_name}. 🤍 Menumbuhkan kembali rasa berharga atau self-worth itu memang tidak terjadi dalam semalam, tetapi sangat mungkin untuk dicapai dengan langkah-langkah kecil yang dilakukan secara konsisten dan penuh kesadaran. Mari kita coba beberapa pendekatan ini:\n\n1.  Setiap hari, luangkan waktu sejenak untuk menuliskan setidaknya satu hal baik tentang dirimu atau satu tindakan positif yang telah kamu lakukan. Ini tidak harus sesuatu yang besar atau spektakuler; hal kecil seperti 'aku berhasil bangun pagi meskipun sulit', 'aku tetap hadir dan mendengarkan temanku bercerita meskipun aku sedang merasa lelah', atau 'aku memberikan senyuman kepada orang asing hari ini' sudah sangat valid dan berarti.\n2.  Belajarlah untuk mengubah narasi atau percakapan internal yang negatif tentang dirimu menjadi kalimat-kalimat yang lebih realistis, suportif, dan berorientasi pada pertumbuhan. Misalnya, jika pikiran 'aku selalu gagal dalam segala hal' muncul, cobalah untuk menggantinya dengan 'aku pernah mengalami kegagalan dalam beberapa hal, tetapi aku juga pernah berhasil dalam hal lain, dan setiap pengalaman adalah kesempatan untuk belajar'.\n3.  Praktikkan ‘mirror check-in’ atau interaksi positif dengan bayangan dirimu di cermin setiap pagi atau malam sebelum tidur. Tatap matamu sendiri, dan ucapkan dengan lembut dan tulus beberapa kalimat afirmasi seperti: 'Aku sedang belajar untuk mengenali dan menerima diriku apa adanya. Aku berharga. Aku cukup.'\n4.  Kurangi atau bahkan tolak standar kesempurnaan yang mustahil, yang mungkin sering kamu lihat di media sosial atau datang dari ekspektasi lingkungan. Jika perlu, berhentilah mengikuti akun-akun atau membatasi interaksi dengan sumber-sumber yang secara konsisten membuatmu merasa tidak cukup atau terus membandingkan diri.\n5.  Jadikan keberanianmu untuk bercerita dan mencari bantuan hari ini sebagai bukti awal bahwa kamu layak untuk merasa lebih baik, kamu sudah cukup berharga, dan kamu sedang dalam proses bertumbuh menjadi versi dirimu yang lebih kuat dan damai. 🌱",
    "Membangun kembali harga diri bukanlah tentang menjadi orang yang paling percaya diri atau tanpa cela di dunia, {user_name}—melainkan tentang belajar untuk mengenali, menerima, dan memeluk setiap versi dirimu, termasuk sisi yang rapuh dan tidak sempurna. Berikut adalah beberapa cara lembut yang bisa kamu mulai terapkan:\n\n1.  Buatlah sebuah ‘jurnal kebaikan diri’ (self-kindness journal). Setiap malam sebelum tidur, tuliskan setidaknya: satu hal kecil yang kamu syukuri dari dirimu atau hidupmu hari itu, satu pelajaran berharga yang kamu dapatkan (meskipun dari kesalahan), dan satu hal yang ingin kamu maafkan dari dirimu sendiri atau dari orang lain terkait hari itu.\n2.  Cobalah untuk mengenali suara siapa sebenarnya di dalam kepalamu yang seringkali melontarkan kritik atau penilaian negatif. Seringkali, itu bukanlah suara hatimu yang sejati, melainkan sisa-sisa dari perkataan atau perlakuan orang lain di masa lalu yang terinternalisasi. Memisahkannya bisa membantu mengurangi bebannya.\n3.  Latihlah toleransi terhadap ketidaksempurnaan dalam dirimu dan dalam apa yang kamu lakukan. Cobalah untuk melakukan aktivitas seperti memasak, menggambar, menulis, atau membuat sesuatu TANPA adanya tuntutan hasil yang harus sempurna. Fokuslah untuk menikmati prosesnya dan ekspresi dirimu.\n4.  Kenali dan rayakan momen-momen kecil yang membuatmu merasa sedikit bangga atau berhasil. Ini tidak harus selalu tentang pencapaian besar; keberanian untuk mencoba hal baru, mengatasi ketakutan kecil, atau sekadar bertahan di hari yang sulit itu juga merupakan kemenangan yang layak diapresiasi.\n5.  Tumbuhkan hubungan yang lebih sehat dan penuh kasih dengan dirimu sendiri. Mulailah dengan menanamkan keyakinan ini dalam hatimu: “Aku tidak harus selalu berprestasi atau menyenangkan semua orang untuk layak dihargai dan dicintai.” Kamu sudah cukup berharga hanya dengan menjadi dirimu apa adanya. 🤲"
  ],
  "self_worth_social_comparison": [
    "Terima kasih sudah bersikap terbuka dan mau mencoba, {user_name}. 🤍 Aku tahu bahwa kebiasaan membandingkan diri dengan orang lain itu bisa sangat melelahkan dan menguras energi, apalagi di era digital seperti sekarang ini di mana kita seolah terus menerus disuguhkan dengan 'kesempurnaan' versi orang lain. Tapi, kamu bisa mulai mengambil langkah-langkah kecil untuk meredakan tekanan ini dan kembali fokus pada dirimu sendiri. Yuk, kita coba beberapa pendekatan ini secara perlahan:\n\n1.  Kurangi paparan terhadap hal-hal yang sering memicu perasaan insecure atau tidak cukup dalam dirimu. Ini bisa berarti membatasi waktu penggunaan media sosial tertentu, berhenti mengikuti (unfollow) atau membisukan (mute) akun-akun yang kontennya sering membuatmu merasa kurang, atau bahkan mengambil jeda total dari platform tertentu untuk sementara waktu. Ingatlah bahwa merawat kesehatan mentalmu juga bisa dimulai dari menjaga apa yang kamu konsumsi secara digital.\n2.  Setiap kali pikiran untuk membandingkan diri muncul ('kenapa aku tidak seperti dia?', 'kenapa dia sudah mencapai ini sedangkan aku belum?'), cobalah untuk secara sadar mengalihkannya dengan pertanyaan yang lebih memberdayakan untuk dirimu sendiri, seperti: 'Apa hal baik atau kemajuan kecil yang sudah aku capai atau miliki hari ini?', atau 'Apa satu langkah kecil yang bisa aku lakukan hari ini untuk mendekati tujuanku sendiri?'. Latihan ini bisa membantumu untuk kembali fokus pada dirimu (grounding) dan menghargai proses serta progresmu sendiri, sekecil apapun itu.\n3.  Buatlah sebuah 'jurnal rasa syukur' atau 'daftar apresiasi diri' harian. Setiap malam sebelum tidur, tuliskan setidaknya tiga hal yang kamu syukuri dari dirimu sendiri atau dari apa yang telah kamu lalui hari itu. Ini tidak harus hal besar; hal kecil seperti berhasil tersenyum tulus kepada orang lain, menyelesaikan satu tugas yang tertunda, atau bahkan sekadar berani untuk bangun dari tempat tidur di hari yang sulit, itu semua adalah hal yang layak untuk disyukuri dan diapresiasi.\n4.  Ingatlah selalu bahwa apa yang kamu lihat di media sosial seringkali hanyalah 'highlight reel' atau potongan-potongan sisi terbaik dari kehidupan seseorang, bukan keseluruhan realita yang kompleks dan penuh tantangan. Orang jarang sekali menunjukkan perjuangan, kegagalan, atau kesedihan mereka secara terbuka. Kamu tidak terlambat atau tertinggal; kamu hanya sedang berada di jalur perjalananmu yang unik, dengan waktu dan prosesmu sendiri.\n5.  Harga diri atau self-worth-mu yang sejati tidaklah ditentukan oleh jumlah likes, status sosial, pencapaian materi, atau validasi dari orang lain. Nilaimu sebagai manusia terletak pada keberanianmu untuk terus bertahan, bertumbuh, belajar, dan menjadi dirimu sendiri apa adanya. 🌱",
    "Membandingkan diri dengan orang lain itu seringkali terasa seperti berjalan di atas treadmill—melelahkan, menguras energi, tapi sebenarnya tidak membawamu maju ke mana-mana. Tapi, kabar baiknya adalah kamu selalu memiliki kekuatan untuk pelan-pelan berhenti dari kebiasaan itu dan mulai fokus pada dirimu sendiri, {user_name}. Mari kita coba mulai dari beberapa langkah ini:\n\n1.  Cobalah untuk menentukan batasan waktu yang jelas untuk bermain media sosial setiap harinya. Misalnya, kamu bisa mengalokasikan maksimal 30 menit hingga 1 jam saja per hari. Sisa waktumu bisa kamu gunakan untuk melakukan hal-hal lain yang lebih nyata, lebih bermakna, dan bisa benar-benar mengisi jiwamu, seperti membaca buku, berolahraga, menekuni hobi, atau berinteraksi langsung dengan orang terdekat.\n2.  Setiap kali kamu melihat pencapaian orang lain dan merasa iri atau minder, cobalah untuk mengubah pola pikirmu. Daripada berpikir 'dia sangat keren, sedangkan aku tidak', cobalah untuk mengatakan pada dirimu sendiri: 'Pencapaiannya memang menginspirasi. Aku juga bisa memiliki versi kesuksesan atau kebahagiaanku sendiri, dengan cara dan jalanku yang unik'. Ingatlah bahwa validasi diri tidak harus selalu datang dari perbandingan dengan orang lain.\n3.  Tuliskan progres atau kemajuan pribadimu setiap minggu, sekecil apapun itu. Misalnya, berhasil mempelajari satu skill baru, menyelesaikan satu bab buku, atau bahkan hanya berhasil menjaga mood tetap stabil selama sehari penuh. Semua itu adalah pencapaian yang nyata dan berharga, meskipun mungkin tidak kamu unggah atau pamerkan di media sosial.\n4.  Carilah atau bangunlah sebuah lingkaran pertemanan atau komunitas yang suportif, bukan yang kompetitif atau sering membuatmu merasa tertekan. Teman-teman atau lingkungan yang bisa memberikan validasi positif, menghargai dirimu apa adanya, dan mendukung pertumbuhanmu jauh lebih berharga daripada mereka yang hanya membuatmu merasa 'ketinggalan' atau tidak cukup.\n5.  Berlatihlah untuk memberikan pujian atau apresiasi kepada dirimu sendiri terlebih dahulu, sebelum kamu memberikannya kepada orang lain. Jika kamu bisa melihat dan mengakui kehebatan atau hal baik pada orang lain, maka kamu juga pasti bisa melihat dan mengakui kehebatan atau hal baik yang ada dalam dirimu sendiri. 💛"
  ],
  "self_worth_imposter_syndrome": [
    "Terima kasih sudah bersedia untuk terbuka dan mencari cara untuk mengatasi perasaan ini, {user_name}. 🤍 Aku tahu bahwa sindrom imposter atau perasaan seperti menjadi 'penipu' itu bisa sangat berat dan mengganggu kepercayaan dirimu. Namun, ada beberapa langkah kecil dan refleksi yang bisa kamu coba terapkan secara perlahan untuk mulai menggeser perspektifmu:\n\n1.  Setiap hari, cobalah untuk menuliskan atau mencatat setidaknya satu pencapaian kecil yang berhasil kamu raih atau satu hal positif yang telah kamu lakukan. Ini tidak harus selalu tentang hal-hal besar atau yang mendapatkan pengakuan dari orang lain; hal-hal seperti berhasil menyelesaikan tugas yang sulit, membantu seseorang, atau bahkan sekadar berani mencoba sesuatu yang baru, itu semua adalah pencapaian yang layak kamu akui dan banggakan.\n2.  Latihlah dirimu untuk mengucapkan afirmasi positif yang realistis dan spesifik mengenai kemampuan serta nilaimu. Misalnya, 'Aku layak berada di sini karena aku telah bekerja keras dan memiliki kontribusi unik', atau 'Aku memiliki kemampuan untuk belajar dan bertumbuh dari setiap pengalaman'. Ulangi kalimat-kalimat ini setiap pagi atau sebelum menghadapi situasi yang memicu perasaan tidak pantasmu.\n3.  Buatlah sebuah 'jurnal refleksi pencapaian'. Setiap kali kamu merasa seperti seorang imposter, tuliskan apa yang memicu perasaan tersebut, lalu coba lawan dengan menuliskan bukti-bukti konkret mengenai usahamu, kemampuanmu, atau kontribusimu yang relevan dengan situasi tersebut. Ini membantu melatih pikiranmu untuk melihat fakta, bukan hanya perasaan.\n4.  Jika memungkinkan, carilah seorang mentor, teman tepercaya, atau role model yang bisa kamu ajak berdiskusi dan yang mungkin juga pernah mengalami perasaan serupa. Mendapatkan perspektif dari orang lain yang bisa melihat kemampuanmu secara objektif seringkali bisa sangat membantu membuka mata dan mengurangi keraguan diri.\n5.  Ingatlah selalu bahwa kamu bukanlah seorang penipu; kamu adalah seorang manusia yang terus belajar, bertumbuh, dan kadang-kadang merasa ragu—dan itu sangatlah wajar. Perasaan tidak pantas tidak selalu sama dengan kenyataan bahwa kamu tidak pantas. 🌱",
    "Perasaan seperti menjadi seorang imposter itu memang bisa membuat kita merasa terjebak dalam keraguan diri dan sulit untuk benar-benar menikmati pencapaian kita, {user_name}. Tapi, kamu bisa mulai mengubah narasi internalmu dengan langkah-langkah kecil yang konsisten. Berikut beberapa pendekatan yang bisa kamu coba:\n\n1.  Setiap kali kamu merasa seperti seorang imposter atau meragukan kelayakanmu, cobalah untuk berhenti sejenak dan tanyakan pada dirimu sendiri dengan jujur: 'Apa bukti konkret yang mendukung pikiran bahwa aku tidak layak atau hanya kebetulan berhasil?' Seringkali, kita akan kesulitan menemukan bukti yang benar-benar kuat untuk mendukung perasaan negatif tersebut.\n2.  Buatlah sebuah daftar yang berisi semua keahlian (skills), kualitas positif, dan pengalaman unik yang kamu miliki. Fokuslah pada apa yang membuatmu berbeda, apa kontribusimu, dan apa nilai tambah yang kamu bawa. Baca kembali daftar ini setiap kali kamu merasa ragu.\n3.  Berhentilah untuk membandingkan dirimu, perjalananmu, atau caramu mencapai sesuatu dengan orang lain. Setiap individu memiliki jalur, tantangan, dan kekuatannya masing-masing yang unik. Fokus pada pertumbuhan dan perkembanganmu sendiri.\n4.  Rayakan setiap kemajuan atau pencapaian yang berhasil kamu raih, sekecil apapun itu. Berikan dirimu sendiri apresiasi dan pengakuan atas usaha keras yang telah kamu lakukan. Kamu berhak untuk merasa bahagia dan bangga atas setiap langkah maju.\n5.  Ingatlah bahwa sindrom imposter adalah fenomena yang umum terjadi, bahkan dialami oleh orang-orang yang sangat kompeten dan berprestasi. Mengalami perasaan ini tidaklah berarti bahwa kamu benar-benar tidak layak atau seorang penipu. Itu lebih merupakan refleksi dari standar tinggi yang mungkin kamu tetapkan untuk dirimu sendiri. 💛"
  ],
  "insomnia_general": [
    "Tentu, {user_name}, aku senang bisa membantumu. Membangun kebiasaan tidur yang sehat memang membutuhkan waktu dan konsistensi, tapi sangat mungkin untuk dicapai. Berikut adalah beberapa tips ringan dan praktis yang mungkin bisa membantumu mendapatkan tidur yang lebih nyenyak dan berkualitas:\n\n1.  Ciptakan sebuah rutinitas relaksasi yang konsisten sebelum tidur. Ini bisa berupa aktivitas seperti membaca buku fisik (hindari layar gadget), melakukan peregangan ringan, menulis jurnal singkat tentang harimu atau apa yang kamu syukuri, atau mendengarkan musik instrumental yang menenangkan. Tujuan utamanya adalah memberikan sinyal kepada tubuh dan pikiranmu bahwa sudah waktunya untuk beristirahat.\n2.  Usahakan untuk menghindari paparan layar HP, laptop, atau televisi setidaknya 30 menit hingga 1 jam sebelum waktu tidur. Cahaya biru yang dipancarkan oleh layar gadget dapat mengganggu produksi melatonin, yaitu hormon alami yang mengatur siklus tidurmu.\n3.  Jika pikiranmu terasa sangat aktif dan sulit untuk tenang, cobalah untuk 'mengosongkan' isi kepalamu dengan menuliskannya semua ke dalam sebuah catatan atau jurnal. Tidak perlu rapi atau terstruktur; cukup keluarkan semua kekhawatiran, ide, atau hal-hal yang mengganggu agar tidak terus berputar di otak.\n4.  Berusahalah untuk konsisten dengan waktu tidur dan waktu bangunmu setiap hari, bahkan di akhir pekan atau hari libur sekalipun. Pola tidur yang teratur dapat membantu mengatur jam biologis tubuhmu menjadi lebih stabil.\n5.  Jangan memaksakan diri untuk tidur jika kamu memang belum merasa mengantuk. Jika setelah 20-30 menit berbaring kamu masih terjaga, cobalah untuk bangun sejenak dan lakukan aktivitas ringan yang menenangkan (seperti membaca atau mendengarkan musik lembut) di ruangan lain hingga rasa kantuk itu datang secara alami. Hindari melakukan aktivitas yang terlalu menstimulasi.\n\nIngatlah, {user_name}, tubuh dan pikiranmu mungkin membutuhkan waktu untuk beradaptasi dan belajar kembali bagaimana cara untuk tenang dan beristirahat dengan baik. Tidak apa-apa untuk menjalaninya secara perlahan dan penuh kesabaran. 🌙",
    "Untuk membantumu mengatasi kesulitan tidur, {user_name}, berikut adalah beberapa tips kecil tambahan yang bisa kamu coba terapkan dalam rutinitas malammu:\n\n1.  Perhatikan suasana di kamar tidurmu. Usahakan untuk membuat kamarmu menjadi tempat yang gelap, sejuk, dan tenang. Redupkan lampu kamar setidaknya satu jam sebelum tidur. Jika perlu, gunakan penutup mata atau penyumbat telinga untuk mengurangi gangguan cahaya dan suara.\n2.  Hindari konsumsi kafein (kopi, teh kental, minuman energi), alkohol, dan makanan berat setidaknya 3-4 jam sebelum waktu tidur. Zat-zat ini bisa mengganggu kualitas tidurmu atau membuatmu sulit untuk terlelap.\n3.  Jika kamu menyukainya, cobalah untuk menggunakan aromaterapi dengan minyak esensial seperti lavender, chamomile, atau sandalwood di kamarmu. Aroma-aroma ini dikenal memiliki efek menenangkan yang bisa membantu relaksasi.\n4.  Latihlah otakmu untuk mengasosiasikan tempat tidur hanya dengan aktivitas tidur dan istirahat. Hindari melakukan aktivitas lain seperti bekerja, belajar, atau bahkan scrolling media sosial di atas kasur. Ini akan membantu membangun sinyal yang kuat bagi otak bahwa kasur adalah tempat untuk beristirahat.\n5.  Cobalah teknik relaksasi otot progresif. Mulai dari ujung kaki, tegangkan otot-ototmu selama beberapa detik, lalu lemaskan sepenuhnya. Lakukan ini secara bertahap ke seluruh bagian tubuh hingga ke area kepala. Teknik ini bisa membantu melepaskan ketegangan fisik yang mungkin kamu tidak sadari. 😴"
  ],
  "heartbreak_breakup": [
    "Kehilangan seseorang yang kamu sayangi karena putus cinta memang bisa terasa seperti kehilangan sebagian besar dari dirimu, {user_name}. Luka ini nyata, dan proses penyembuhannya membutuhkan waktu serta kesabaran. Tapi, kamu tidak sendirian dalam perjalanan ini. Yuk, mari kita coba beberapa langkah perlahan untuk merawat hatimu dan membantumu menemukan kembali pijakanmu:\n\n1.  Validasi setiap perasaan yang muncul. Sangat wajar jika kamu merasa sedih, marah, kecewa, bingung, atau bahkan mati rasa. Izinkan dirimu untuk merasakan semua itu tanpa menghakimi. Mengakui perasaan adalah langkah awal yang penting menuju penyembuhan.\n2.  Jika memungkinkan, berikan jarak sementara dari hal-hal yang bisa memicu kenangan atau rasa sakit berlebih. Ini bisa berarti menghapus atau menyembunyikan sementara kenangan digital (foto, chat), atau menghindari tempat-tempat tertentu. Ini bukan berarti melarikan diri, tetapi memberikan ruang bagi hatimu untuk bernapas.\n3.  Jangan pernah menuntut dirimu untuk ‘harus cepat sembuh’ atau ‘segera move on’. Setiap individu memiliki waktu dan cara pemulihannya masing-masing. Tidak ada standar yang baku. Hargai ritmemu sendiri.\n4.  Cobalah untuk menciptakan momen-momen baru atau rutinitas baru yang bisa memberikan sedikit kesegaran dalam harimu. Ini bisa berupa hal sederhana seperti mencoba resep masakan baru, berjalan-jalan di taman yang belum pernah kamu kunjungi, atau memulai kembali hobi lama yang sempat terlupakan.\n5.  Ingatlah bahwa setiap hari di mana kamu berhasil berdiri, bernapas, dan melanjutkan hidup, meskipun dengan hati yang masih terasa berat, itu adalah sebuah bentuk kemenangan kecil. Dan setiap kemenangan kecil itu sangatlah layak untuk kamu rayakan dan apresiasi. 🤍",
    "Patah hati karena putus cinta memang tidak pernah mudah, {user_name}, dan seringkali meninggalkan luka yang tidak terlihat namun begitu terasa dampaknya hingga ke helaan napas. Kamu sudah sangat hebat karena mau mengakui rasa sakitmu dan mencari cara untuk pulih. Yuk, mari kita coba bersama-sama merawat hati yang sedang terluka ini dengan beberapa pendekatan lembut:\n\n1.  Buatlah sebuah playlist lagu-lagu yang bisa mendukung proses ‘healing’ atau penyembuhanmu. Ini tidak harus selalu lagu sedih untuk meluapkan perasaan, tetapi juga bisa lagu-lagu yang menenangkan, membangkitkan semangat, atau mengingatkanmu akan kekuatan dirimu sendiri.\n2.  Latihlah afirmasi positif harian yang bisa menguatkanmu. Ucapkan dengan tulus kepada dirimu sendiri kalimat-kalimat seperti: ‘Aku sudah cukup baik apa adanya. Aku layak untuk dicintai dan bahagia. Aku akan melewati ini.’\n3.  Setiap malam sebelum tidur, cobalah untuk menuliskan setidaknya tiga hal positif tentang dirimu sendiri atau tiga hal yang berhasil kamu syukuri hari itu, sekecil apapun. Latihan ini bisa membantu membangun kembali harga dirimu dan mengalihkan fokus dari rasa kehilangan.\n4.  Jangan merasa bersalah atau lemah jika kamu masih sering merasa kangen atau teringat kenangan bersamanya. Itu adalah hal yang sangat manusiawi, bukan sebuah kelemahan. Berikan dirimu waktu untuk memproses semua itu.\n5.  Setiap langkah kecil yang kamu ambil untuk maju, seperti berhasil tidak menghubunginya hari ini atau berhasil tersenyum tulus, itu adalah sebuah progres yang berarti. Bahkan jika kamu harus melaluinya sambil menangis sesekali, itu tidak apa-apa. 🌦️"
  ],
  "heartbreak_cheated": [
    "Terima kasih sudah bersedia untuk terbuka dan mencari cara untuk pulih, {user_name}. Luka karena diselingkuhi itu memang tidak mudah untuk hilang begitu saja, tapi kamu memiliki kekuatan untuk memulai proses penyembuhan secara perlahan. Ini beberapa langkah yang bisa kamu coba untuk merawat dirimu:\n\n1.  Hentikan kebiasaan untuk menyalahkan dirimu sendiri atas apa yang telah terjadi. Ingatlah bahwa keputusan untuk berselingkuh adalah sepenuhnya tanggung jawabnya, bukan karena ada kekurangan dalam dirimu. Kamu tidak salah karena telah percaya dan mencintai.\n2.  Berikan dirimu sendiri jarak dan waktu yang cukup dari mantan pasanganmu. Jika memungkinkan, putus kontak untuk sementara waktu (misalnya, dengan memblokir atau menyembunyikan akun media sosialnya). Ini bukan berarti kamu lemah atau lari dari masalah, tetapi ini adalah bentuk perlindungan diri yang penting agar hatimu memiliki ruang untuk sembuh tanpa terus menerus terpicu.\n3.  Tuliskan semua perasaanmu dalam sebuah jurnal atau catatan pribadi. Ungkapkan semua rasa marah, kecewa, sedih, bingung, atau bahkan rasa sakit fisik yang mungkin kamu rasakan. Biarkan semua emosi itu keluar tanpa sensor. Menulis bisa menjadi cara yang sangat melegakan.\n4.  Kelilingi dirimu dengan support system yang positif dan suportif. Carilah teman, anggota keluarga, atau bahkan komunitas yang bisa mendengarkan ceritamu tanpa menghakimi, memberikanmu dukungan emosional, dan mengingatkanmu akan nilai dirimu.\n5.  Ingatlah selalu bahwa proses menyembuhkan diri dari pengkhianatan ini bukanlah tentang ‘melupakan’ apa yang telah terjadi, melainkan tentang memulihkan kendali atas hatimu sendiri, membangun kembali kepercayaan pada dirimu, dan menemukan kembali kebahagiaanmu. 🌱",
    "Aku tahu bahwa ini adalah situasi yang sangat berat dan menyakitkan, {user_name}, tetapi kamu sudah memulai langkah yang sangat penting dengan mau mencari cara untuk bangkit dan pulih. Yuk, kita coba beberapa pendekatan ini bersama-sama:\n\n1.  Validasi setiap perasaan yang muncul dalam dirimu. Semua rasa—marah yang membara, kekecewaan yang mendalam, kesedihan yang tak tertahankan—itu semua adalah reaksi yang sangat wajar dan manusiawi setelah mengalami pengkhianatan. Jangan menekan atau meremehkan perasaanmu sendiri.\n2.  Batasi aksesmu terhadap hal-hal yang bisa mengingatkanmu padanya atau memicu kembali rasa sakit. Ini bisa termasuk tidak lagi melihat media sosialnya, menghapus foto-foto bersama, atau menyimpan barang-barang pemberiannya di tempat yang tidak mudah terlihat untuk sementara waktu.\n3.  Fokuslah untuk membangun kembali rutinitas kecil yang bisa memberikan rasa stabilitas dan kenyamanan dalam keseharianmu. Hal-hal sederhana seperti bangun pagi, mandi, makan teratur, dan tidur yang cukup itu juga merupakan bentuk penyembuhan diri yang penting.\n4.  Buatlah sebuah 'healing box' atau 'kotak pemulihan' pribadi. Isi kotak tersebut dengan hal-hal yang bisa membuatmu merasa nyaman, tenang, atau terhibur—misalnya, buku favorit, aromaterapi, cokelat, foto orang-orang tersayang, atau daftar lagu yang menenangkan.\n5.  Ingatlah bahwa kamu bukanlah pribadi yang rusak atau cacat karena telah diselingkuhi. Kamu sedang dalam proses pemulihan dari sebuah luka yang dalam. Dan setiap proses penyembuhan itu valid, unik, dan membutuhkan waktu. 💛"
  ],
  "heartbreak_rejected": [
    "Aku memahami betapa beratnya perasaanmu setelah mengalami penolakan, {user_name}. Sangat wajar jika kamu merasa sedih, kecewa, atau bahkan mempertanyakan banyak hal tentang dirimu sendiri. Tapi, ingatlah bahwa penolakan ini tidak mengurangi nilaimu sebagai individu. Yuk, kita coba beberapa langkah perlahan untuk membantumu memproses dan menyembuhkan luka ini:\n\n1.  Akui dan validasi setiap perasaan yang muncul. Jangan menekan atau menyangkal rasa sakitmu. Izinkan dirimu untuk merasa sedih, marah, atau kecewa. Memberi ruang bagi emosi adalah langkah awal yang penting untuk penyembuhan.\n2.  Hindari menyalahkan dirimu sendiri atau mencari-cari kekurangan dalam dirimu sebagai alasan penolakan. Ingatlah bahwa ketidakcocokan atau keputusan orang lain seringkali bukan cerminan dari nilaimu. Kamu tetap berharga apa adanya.\n3.  Alihkan fokus dan energimu pada hal-hal yang bisa membangun dirimu dan memberikanmu kebahagiaan. Ini bisa berupa menekuni hobi yang kamu sukai, belajar hal baru, berolahraga, atau menghabiskan waktu berkualitas dengan orang-orang yang mendukungmu.\n4.  Kurangi kebiasaan untuk terus menerus memikirkan atau 'mengulang' kejadian penolakan tersebut di kepalamu. Jika pikiran itu muncul, cobalah untuk secara sadar mengalihkannya ke hal lain, atau lakukan teknik grounding seperti fokus pada napas atau sensasi di sekitarmu.\n5.  Jika kamu merasa nyaman, cobalah untuk menuliskan perasaanmu dalam sebuah jurnal atau catatan. Terkadang, menuangkan isi hati melalui tulisan bisa membantu melepaskan beban emosional dan memberikan perspektif baru. Kamu juga bisa membayangkan dirimu di masa depan yang sudah berhasil melewati ini dan merasa lebih kuat. 🌷",
    "Di balik rasa sakit dan kekecewaan akibat penolakan ini, {user_name}, selalu ada ruang bagi dirimu untuk bertumbuh menjadi pribadi yang lebih kuat dan bijaksana. Jika kamu merasa siap untuk memulai proses penyembuhan, mari kita coba beberapa pendekatan ini secara perlahan:\n\n1.  Berikan pertanyaan reflektif kepada dirimu sendiri: 'Apa yang sebenarnya aku cari atau harapkan dari hubungan atau interaksi ini?' Memahami kebutuhan dan harapanmu sendiri bisa membantu mengurangi ketergantungan pada validasi eksternal.\n2.  Jangan pernah membandingkan perjalanan cintamu atau pengalaman penolakanmu dengan orang lain. Setiap individu memiliki waktu, proses, dan jalan ceritanya masing-masing. Fokuslah pada penyembuhan dan pertumbuhanmu sendiri.\n3.  Cobalah untuk melakukan 'detoks hati' setidaknya selama beberapa hari atau minggu ke depan. Ini bisa berarti tidak lagi memeriksa media sosial orang yang menolakmu, tidak membuka kembali percakapan lama, dan tidak terus menerus menyalahkan diri sendiri atau dirinya.\n4.  Perkuat koneksi dengan dirimu sendiri melalui aktivitas yang menenangkan dan memberdayakan. Ini bisa melalui meditasi singkat setiap hari, menuliskan afirmasi positif tentang dirimu, atau melakukan percakapan yang suportif dengan dirimu sendiri di depan cermin.\n5.  Rayakan setiap langkah kecil dalam proses penyembuhanmu. Misalnya, jika hari ini kamu berhasil untuk tidak memikirkannya selama beberapa jam, atau jika kamu berhasil melakukan aktivitas yang menyenangkan tanpa terbebani kesedihan, berikan apresiasi untuk dirimu sendiri. Itu adalah progres yang berarti. 🧡"
  ],
  "heartbreak_ghosted": [
    "Diabaikan atau ditinggalkan tanpa alasan yang jelas itu memang bisa membuat hati terasa bingung, kecewa, dan sangat sakit, {user_name}. Tapi, ketahuilah bahwa kamu tetaplah pribadi yang utuh dan layak untuk mendapatkan cinta serta perlakuan yang penuh penghargaan. Yuk, mari kita coba beberapa langkah kecil ini untuk membantumu merawat diri dan memproses pengalaman ini:\n\n1.  Berhentilah untuk mencari-cari alasan atau penjelasan dari dirinya. Terkadang, kita tidak akan pernah tahu dengan pasti mengapa seseorang melakukan ghosting, dan itu bukanlah sepenuhnya salahmu atau tanggung jawabmu untuk menemukan jawabannya. Fokuslah pada apa yang bisa kamu kontrol saat ini, yaitu dirimu sendiri.\n2.  Jika kamu masih sering tergoda untuk memeriksa media sosialnya atau menunggu kabarnya, cobalah untuk memblokir atau setidaknya menyembunyikan sementara kontak atau akunnya. Ini bukanlah tindakan yang kekanak-kanakan, melainkan sebuah bentuk perlindungan diri (self-protection) yang penting agar hatimu memiliki ruang untuk tenang dan sembuh.\n3.  Buatlah sebuah playlist lagu-lagu penyembuhan. Pilihlah lagu-lagu yang liriknya bisa membuatmu merasa dipahami, tidak sendirian, atau bahkan bisa membangkitkan semangatmu kembali. Musik bisa menjadi teman yang baik dalam proses ini.\n4.  Setiap hari, cobalah untuk menuliskan atau mengingat setidaknya satu hal yang kamu banggakan dari dirimu sendiri, atau satu hal kecil yang berhasil kamu lakukan dengan baik hari itu. Sekecil apapun, apresiasi diri ini penting untuk membangun kembali rasa berhargamu.\n5.  Ingatlah selalu bahwa ghosting itu lebih mencerminkan karakter dan ketidakmampuan orang yang melakukannya untuk berkomunikasi dengan baik, bukan cerminan dari nilai atau kelayakan dirimu untuk dicintai. Kamu tidak kurang, dia yang mungkin belum cukup dewasa atau berani untuk jujur. 💔",
    "Ghosting itu memang bisa meninggalkan luka yang dalam karena kita seringkali ditinggalkan dengan begitu banyak pertanyaan yang belum terjawab dan rasa bingung yang tak berkesudahan, {user_name}. Tapi, kamu berhak untuk melepaskan diri dari lingkaran overthinking ini dan memulai proses pemulihanmu. Berikut adalah beberapa langkah praktis yang bisa kamu coba:\n\n1.  Jika kamu masih menyimpan percakapan terakhir dengannya, cobalah untuk menuliskan semua unek-unek atau pertanyaan yang ingin kamu sampaikan kepadanya dalam sebuah catatan pribadi (yang tidak perlu dikirim), lalu setelah itu hapus atau arsipkan percakapan tersebut. Tindakan simbolis ini terkadang bisa memberikan sedikit rasa lega dan penutupan.\n2.  Buatlah sebuah batasan waktu yang jelas untuk dirimu sendiri dalam merasakan kesedihan atau kekecewaan ini. Misalnya, izinkan dirimu untuk merasa sedih atau marah hingga akhir pekan ini, tetapi setelah itu berjanjilah pada dirimu sendiri untuk mulai mengambil langkah-langkah kecil untuk bangkit dan fokus pada hal lain.\n3.  Lakukan refleksi diri: 'Pelajaran berharga apa yang bisa aku ambil dari pengalaman ini mengenai diriku sendiri, kebutuhanku dalam sebuah hubungan, atau tipe orang yang sebaiknya aku hindari di masa depan?'\n4.  Cobalah untuk menciptakan atau kembali ke rutinitas pagi yang baru dan menyegarkan. Misalnya, bangun lebih awal dari biasanya, minum segelas air putih hangat, melakukan peregangan ringan, dan mengucapkan afirmasi positif seperti 'Aku sudah cukup baik apa adanya dan aku siap menjalani hari ini dengan semangat baru'.\n5.  Ingatlah bahwa kamu bukanlah masalahnya di sini—kamu mungkin adalah korban dari ketidakjelasan atau ketidakdewasaan seseorang yang bukan kamu yang ciptakan atau sebabkan. Fokuslah pada penyembuhan dirimu. 🤍"
  ],
  "loneliness_no_friends": [
    "Aku sungguh mengerti keinginanmu untuk terhubung dan tidak merasa sendirian lagi, {user_name}. Rasa kesepian karena tidak memiliki teman dekat itu bisa sangat sunyi dan memberatkan. Tapi, ketahuilah bahwa kamu tidak sendirian dalam perasaan ini, dan selalu ada cara untuk mulai membangun koneksi baru secara perlahan dan penuh kesadaran. Berikut adalah beberapa langkah yang bisa kamu coba:\n\n1.  Cobalah untuk menemukan atau bergabung dengan komunitas yang berbasis minat atau hobimu, baik itu secara online maupun offline jika memungkinkan. Kamu bisa mencari grup diskusi tentang buku yang kamu suka, komunitas musik, kelompok pecinta game, atau bahkan kelas seni. Seringkali, teman-teman terbaik justru datang dari kesamaan minat atau aktivitas yang sama-sama kita nikmati, karena itu bisa menjadi jembatan alami untuk memulai percakapan.\n2.  Mulailah dari langkah kecil dengan mencoba menyapa atau menghubungi kembali orang-orang terdekat yang mungkin sudah lama tidak berinteraksi denganmu. Ini bisa teman lama dari sekolah atau kuliah, dosen yang pernah suportif, atau bahkan kenalan yang kamu rasa memiliki energi positif. Tidak perlu langsung bercerita panjang lebar; cukup dengan mengirim pesan singkat menanyakan kabar atau mengucapkan salam bisa menjadi awal yang baik untuk kembali terkoneksi.\n3.  Pertimbangkan untuk mengikuti kegiatan sukarela (volunteering) atau proyek kolaboratif di lingkungan sekitarmu atau secara online. Di tempat-tempat seperti ini, kamu memiliki kesempatan untuk bertemu dengan orang-orang baru yang memiliki kepedulian atau tujuan yang sama, dan interaksi seringkali terjadi secara lebih alami melalui kerja sama.\n4.  Latihlah keberanianmu dalam berinteraksi melalui tindakan-tindakan kecil sehari-hari. Misalnya, cobalah untuk memberikan senyuman tulus kepada penjaga kantin atau kasir, menyapa teman sekelas atau rekan kerja yang jarang kamu ajak bicara, atau bahkan hanya sekadar memberikan komentar yang positif dan suportif di unggahan media sosial seseorang yang kamu kagumi.\n5.  Berikan apresiasi dan penghargaan kepada dirimu sendiri setiap kali kamu berhasil mencoba untuk keluar dari zona nyamanmu atau melakukan satu langkah kecil untuk terhubung dengan orang lain, terlepas dari apapun hasilnya. Ingatlah bahwa koneksi yang tulus dan bermakna itu tumbuh dari konsistensi, keberanian untuk mencoba, dan kesabaran dalam berproses. 🌱",
    "Rasa sendirian itu memang bisa membuat hati terasa sangat berat, {user_name}, apalagi jika kamu merasa tidak memiliki teman yang bisa diajak berbagi. Tapi, kamu tidak harus melewati perasaan ini sendirian. Berikut adalah beberapa langkah atau pendekatan yang mungkin bisa membantumu untuk mulai membangun koneksi baru:\n\n1.  Jika kamu adalah seorang pelajar atau mahasiswa, cobalah untuk lebih aktif terlibat dalam kegiatan atau organisasi kampus, atau bergabung dengan kelas-kelas online yang sesuai dengan minatmu. Tempat-tempat seperti ini seringkali menjadi lingkungan yang aman dan kondusif untuk mulai berlatih berinteraksi dan bertemu dengan orang-orang baru secara perlahan.\n2.  Ikutilah acara-acara sosial, seminar, webinar, atau workshop yang topiknya sesuai dengan minat atau hobimu. Selain mendapatkan ilmu atau pengalaman baru, ini juga bisa menjadi titik awal yang baik untuk menemukan teman ngobrol atau orang-orang yang memiliki ketertarikan serupa denganmu.\n3.  Tantang dirimu sendiri untuk mengajukan pertanyaan-pertanyaan sederhana kepada orang-orang di sekitarmu, terutama jika ada kesempatan. Misalnya, kamu bisa bertanya kepada teman sekelas, 'Kamu suka genre film apa untuk ditonton di akhir pekan?' atau kepada rekan kerja, 'Apakah kamu pernah mencoba tempat makan baru yang ada di dekat sini?'. Percakapan ringan bisa membuka pintu ke interaksi yang lebih dalam.\n4.  Jika kamu memiliki sebuah proyek pribadi atau hobi yang sedang kamu tekuni, cobalah untuk membagikan proses atau hasilnya secara online, misalnya melalui media sosial, blog, atau forum diskusi. Terkadang, berbagi minat seperti ini bisa membantu kamu untuk terhubung dengan orang lain yang memiliki ketertarikan serupa, tanpa harus selalu memulai obrolan secara langsung terlebih dahulu.\n5.  Selalu ingat dan tanamkan dalam hatimu bahwa kamu pantas untuk didengar, dimengerti, dan ditemani. Yuk, kita coba mulai dari membangun satu koneksi baru saja dulu, sekecil apapun itu. 🤝"
  ],
  "loneliness_no_one_to_talk_to": [
    "Aku sungguh mendengar dan memahami keinginanmu untuk memiliki tempat berbagi yang aman dan suportif, {user_name}. Rasa tidak memiliki siapa pun untuk diajak bicara atau mencurahkan isi hati itu bisa membuat kita merasa sangat terisolasi dan beban di dada terasa semakin berat. Tapi, ketahuilah bahwa kamu tidak sendirian dalam perasaan ini, dan selalu ada cara untuk mulai membangun koneksi baru secara perlahan dan penuh kesadaran. Berikut adalah beberapa tips yang mungkin bisa membantumu:\n\n1.  Cobalah untuk memulai dengan menuliskan semua perasaan dan pikiranmu dalam sebuah jurnal pribadi. Ini bisa menjadi 'teman' pertama yang selalu siap mendengarkan tanpa menghakimi, dan seringkali proses menulis itu sendiri bisa membantu melepaskan sebagian beban emosionalmu.\n2.  Jika kamu merasa nyaman, cobalah untuk menggunakan fitur pesan suara (voice note) untuk 'berbicara' kepada dirimu sendiri atau seolah-olah kamu sedang bercerita kepada seseorang. Mendengarkan kembali suaramu sendiri saat mengungkapkan perasaan terkadang bisa membantu memproses emosi dengan lebih dalam dan memberikan perspektif baru.\n3.  Pertimbangkan untuk mencari dan bergabung dengan platform curhat anonim yang aman dan terpercaya, seperti Vent, 7 Cups, atau forum-forum online lain yang memiliki moderator. Di sana, kamu bisa berbagi cerita atau perasaanmu tanpa harus mengungkapkan identitas aslimu, dan seringkali bisa mendapatkan dukungan dari orang-orang yang mengalami hal serupa.\n4.  Mulailah untuk membangun kembali atau memperkuat koneksi-koneksi mikro dalam kehidupan sehari-harimu. Ini bisa berupa tindakan kecil seperti menyapa penjaga kos atau tetangga dengan ramah, mengirimkan pesan singkat untuk menanyakan kabar kepada teman lama yang sudah jarang berinteraksi, atau bahkan hanya sekadar memberikan reaksi emoji yang positif di unggahan media sosial seseorang yang kamu kenal.\n5.  Jika kamu merasa beban ini terlalu berat untuk ditanggung sendirian dan kamu memiliki akses, jangan ragu untuk mempertimbangkan mencari bantuan dari tenaga profesional seperti konselor atau psikolog. Terkadang, seorang profesional bisa menjadi pendengar yang objektif, memberikan dukungan yang tepat, dan membantumu menemukan jalan keluar atau cara mengatasi masalahmu. Ingatlah, mencari bantuan adalah tanda kekuatan, bukan kelemahan. 🌿",
    "Aku tahu betapa beratnya ketika kita merasa seolah tidak ada seorang pun yang bisa menjadi tempat kita untuk bercerita atau berbagi beban, {user_name}. Tapi, kamu tetaplah pribadi yang berharga, dan perasaan serta ceritamu sangatlah layak untuk didengarkan dan dipahami. Mari kita coba beberapa pendekatan ini untuk membantumu merasa sedikit lebih terhubung:\n\n1.  Cobalah untuk lebih mengenali pola kesendirian atau perasaan tidak memiliki tempat curhat yang kamu alami. Apakah ada waktu-waktu tertentu di mana perasaan ini lebih intens muncul? Apakah ada pemicu khusus yang membuatnya semakin kuat? Dengan memahami polanya, kamu bisa meresponsnya dengan lebih sadar dan penuh kasih sayang terhadap dirimu sendiri.\n2.  Jika kamu tertarik, cobalah untuk mencari dan bergabung dengan sesi 'sharing circle' atau kelompok dukungan online yang membahas topik-topik yang relevan dengan apa yang kamu rasakan. Banyak komunitas yang menciptakan ruang aman (safe space) di mana anggotanya bisa berbagi pengalaman dan saling mendukung tanpa syarat.\n3.  Beranikan dirimu untuk menyapa atau memulai interaksi ringan dengan setidaknya satu orang baru atau orang yang jarang kamu ajak bicara setiap minggunya. Tidak perlu langsung curhat atau bercerita hal yang berat; cukup dengan menunjukkan keberadaanmu, menanyakan kabar, atau membahas topik netral bisa menjadi awal yang baik.\n4.  Latihlah dirimu untuk melakukan praktik rasa syukur (gratitude) harian. Setiap hari, cobalah untuk menuliskan atau mengingat setidaknya tiga hal kecil yang masih bisa kamu syukuri kehadirannya dalam hidupmu. Fokus pada hal-hal positif ini, meskipun terasa sulit, bisa membantu membuka hatimu dan pikiranmu terhadap kemungkinan-kemungkinan baru.\n5.  Selalu ingat dan tanamkan dalam hatimu bahwa kamu tidak harus menjadi pribadi yang super sosial atau ekstrovert untuk bisa memiliki koneksi yang bermakna. Terkadang, cukup dengan terhubung secara tulus dan mendalam dengan satu atau dua orang saja itu sudah bisa membuat perbedaan besar dalam hidupmu. 🫂"
  ],
  "grief_loss_of_person": [
    "Terima kasih sudah bersedia untuk terbuka dan mencari cara untuk merawat dirimu di tengah duka ini, {user_name}. 🤍 Kehilangan orang tersayang adalah salah satu luka batin terdalam, dan proses penyembuhannya membutuhkan waktu, kesabaran, serta kasih sayang yang besar terhadap diri sendiri. Kamu tidak sendirian, dan kamu tidak harus merasa kuat setiap saat. Yuk, mari kita coba beberapa langkah kecil ini bersama-sama untuk membantumu melewati masa sulit ini:\n\n1.  Izinkan dirimu untuk merasakan setiap emosi yang muncul. Sangat wajar jika kamu merasa sedih yang mendalam, marah, bingung, hampa, atau bahkan mati rasa. Jangan menekan atau menyangkal perasaanmu; akui keberadaannya dan berikan ruang bagi dirimu untuk menangis jika memang itu yang kamu butuhkan.\n2.  Cobalah untuk menuliskan kenangan-kenangan indah atau hal-hal baik yang kamu ingat tentang orang yang telah pergi. Aktivitas ini bisa menjadi bentuk penghormatan yang tulus dan juga cara yang menenangkan untuk memproses rasa kehilanganmu. Fokus pada cinta dan kebaikan yang pernah ada.\n3.  Bangun kembali atau pertahankan rutinitas harian yang ringan dan bisa memberimu sedikit struktur. Hal-hal sederhana seperti mandi tepat waktu, makan teratur, atau berjalan kaki singkat di pagi hari bisa membantu mengurangi perasaan hampa dan memberikan sedikit rasa normalitas.\n4.  Bicaralah dengan orang-orang yang kamu percaya dan bisa memberikan dukungan emosional tanpa menghakimi. Ini bisa teman dekat, anggota keluarga, atau bahkan seorang konselor. Berbagi cerita atau sekadar didengarkan seringkali bisa meringankan beban di hatimu.\n5.  Ingatlah bahwa proses berduka itu unik untuk setiap individu. Tidak ada cara yang benar atau salah, dan tidak ada batasan waktu yang pasti. Bersabarlah dengan dirimu sendiri dan hargai setiap langkah kecil dalam perjalanan penyembuhanmu.",
    "Kehilangan seseorang yang kita cintai memang tidak harus langsung ‘dilampaui’ atau dilupakan begitu saja, {user_name}. Proses berduka adalah sebuah perjalanan yang bisa penuh dengan gelombang emosi. Namun, ada beberapa langkah kecil dan penuh kasih yang bisa kamu lakukan untuk membantu dirimu tetap berdiri dan merawat hatimu di tengah situasi ini:\n\n1.  Setiap kali rasa sesak atau kesedihan yang mendalam itu datang, cobalah untuk berhenti sejenak dan bernapas dengan perlahan dan dalam. Sambil melakukannya, kamu bisa mengucapkan afirmasi sederhana dalam hati, seperti: ‘Aku tidak sendirian dalam rasa ini. Aku sedang dalam proses menyembuhkan diriku. Aku kuat.’\n2.  Kurangi paparan terhadap hal-hal atau situasi yang mungkin bisa memicu luka atau kesedihanmu secara berlebihan untuk sementara waktu. Tidak apa-apa untuk mengambil jarak dari media sosial, tempat-tempat tertentu, atau bahkan percakapan yang terasa terlalu berat jika kamu belum siap.\n3.  Jangan pernah memaksakan dirimu untuk ‘cepat move on’ atau segera melupakan semuanya. Proses penyembuhan dari rasa duka itu bukanlah sebuah perlombaan. Setiap orang memiliki ritme dan caranya sendiri. Hargai prosesmu.\n4.  Jika kamu merasa nyaman, cobalah untuk menulis sebuah surat yang ditujukan kepada orang yang telah pergi. Dalam surat itu, kamu bisa mengungkapkan semua hal yang belum sempat kamu katakan, rasa rindu, cinta, atau bahkan penyesalanmu. Ini bisa menjadi cara yang sangat terapeutik untuk melepaskan emosi yang terpendam.\n5.  Pertimbangkan untuk mencari dukungan dari tenaga profesional seperti konselor atau psikolog jika kamu merasa beban ini terlalu berat untuk dihadapi sendirian. Mencari bantuan adalah tanda kekuatan dan kepedulian terhadap dirimu sendiri, bukan sebuah kelemahan. 🌱"
  ],
  "grief_loss_of_pet": [
    "Kehilangan hewan peliharaan memang bisa terasa seperti kehilangan seorang sahabat sejati atau bahkan anggota keluarga, {user_name}. Luka dan kesedihan yang kamu rasakan sangatlah nyata dan valid. Yuk, mari kita coba beberapa langkah kecil ini untuk membantumu merawat hatimu dan memulai proses pemulihan secara perlahan:\n\n1.  Izinkan dirimu untuk berduka sepenuhnya. Sangat wajar jika kamu merasa sedih, menangis, merasa hampa, atau bahkan marah. Jangan menekan atau menyangkal perasaanmu; berikan ruang bagi dirimu untuk merasakan setiap emosi yang muncul tanpa penghakiman. Kamu tidak perlu terburu-buru untuk merasa 'baik-baik saja'.\n2.  Kenanglah momen-momen bahagia dan indah yang pernah kamu lalui bersama hewan peliharaanmu. Kamu bisa menuliskan kenangan-kenangan tersebut dalam sebuah jurnal, membuat album foto digital, atau bahkan sekadar meluangkan waktu untuk mengingat kembali saat-saat menyenangkan bersamanya. Ini bisa membantu hatimu untuk fokus pada cinta dan kebahagiaan yang pernah ada, bukan hanya pada rasa kehilangan.\n3.  Jika kamu merasa nyaman, cobalah untuk membuat sebuah ritual kecil sebagai bentuk perpisahan atau penghormatan terakhir untuknya. Ini bisa berupa menyalakan lilin sambil mengenangnya, menanam bunga atau pohon di tempat favoritnya, atau menuliskan surat perpisahan yang berisi semua perasaanmu untuknya.\n4.  Jaga rutinitas harianmu sebisa mungkin. Meskipun mungkin terasa berat, usahakan untuk tetap bangun pagi, makan secara teratur, dan menjaga kebersihan diri. Hal-hal kecil ini bisa membantu memberikan sedikit struktur dan stabilitas di tengah perasaan duka yang mungkin membuat segalanya terasa tidak menentu.\n5.  Bicaralah atau berbagi cerita tentang hewan peliharaanmu dan perasaan kehilanganmu kepada orang-orang yang kamu percaya dan bisa memahami rasa sakit ini. Ini bisa teman dekat, anggota keluarga, atau bahkan komunitas pecinta hewan online yang mungkin pernah mengalami hal serupa. Mengetahui bahwa kamu tidak sendirian bisa sangat menguatkan. 🐾",
    "Aku tahu bahwa rasa kehilangan hewan kesayanganmu ini bisa sangat sunyi dan menyakitkan, {user_name}. Tapi, ketahuilah bahwa kamu tidak harus melewati proses berduka ini sendirian. Berikut adalah beberapa pendekatan lembut yang bisa kamu coba untuk merawat dirimu:\n\n1.  Buatlah sebuah waktu khusus setiap hari untuk mengenang hewan peliharaanmu dengan cara yang positif. Misalnya, setiap malam sebelum tidur, kamu bisa meluangkan waktu 5-10 menit untuk mengingat kembali satu hal lucu, manis, atau baik yang pernah ia lakukan dan membuatmu tersenyum.\n2.  Tuliskan semua isi hati dan perasaanmu dalam sebuah jurnal. Biarkan semua emosi yang belum sempat keluar atau terungkap bisa memiliki tempat untuk diekspresikan. Kamu bisa menulis tentang rasa rindumu, kenangan bersamanya, atau bahkan rasa bersalah jika memang ada. Proses menulis ini bisa sangat melegakan.\n3.  Berikan nama pada setiap rasa atau emosi yang kamu alami. Apakah itu rasa sedih, marah, hampa, kecewa, atau bahkan rasa bersalah? Dengan mengakui dan memberi nama pada emosi tersebut, kamu bisa mulai untuk memahaminya dan berdamai dengannya secara perlahan.\n4.  Lakukan aktivitas perawatan diri (self-care) secara intensif. Ini bisa berupa makan makanan yang hangat dan bergizi, memastikan kamu mendapatkan tidur yang cukup (meskipun mungkin sulit), atau mengurangi paparan terhadap konten-konten di media sosial yang mungkin bisa memicu kesedihanmu secara berlebihan.\n5.  Ingatlah selalu bahwa cinta dan kasih sayang yang pernah kamu berikan kepadanya, dan yang pernah ia berikan kepadamu, itu akan tetap hidup di dalam hatimu dan menjadi kenangan yang berharga. Dan cinta itu tidak akan pernah hilang begitu saja. 🐶🐱"
  ],
  "grief_due_to_divorce": [
    "Aku memahami bahwa perceraian orang tua bisa terasa seperti bumi di bawah kakimu bergeser dan meninggalkan banyak ketidakpastian, {user_name}. Situasi ini memang tidak mudah, tetapi ada beberapa hal kecil dan penuh kasih yang bisa kamu lakukan untuk membantu menstabilkan dirimu dan memulai proses penyembuhan:\n\n1.  Akui dan terimalah bahwa hidupmu sedang mengalami perubahan besar. Kamu tidak harus langsung mengerti atau menerima semuanya saat ini juga. Mulailah dengan memberikan dirimu sendiri izin untuk merasakan setiap emosi yang muncul, entah itu sedih, marah, bingung, atau bahkan mati rasa. Semua itu valid.\n2.  Ciptakan sebuah 'zona aman' atau 'ruang tenang' untuk dirimu sendiri. Ini bisa berupa sudut kamarmu yang nyaman, sebuah taman yang teduh, atau tempat lain di mana kamu bisa merasa aman untuk sekadar duduk diam, menangis jika perlu, menulis jurnal, atau hanya untuk bernapas dan menenangkan diri tanpa gangguan.\n3.  Cobalah untuk membangun kembali atau mempertahankan rutinitas-rutinitas kecil yang bisa memberikan sedikit struktur dan prediktabilitas dalam keseharianmu. Misalnya, sarapan sendiri dengan mendengarkan lagu kesukaanmu, meluangkan waktu untuk menyiram tanaman setiap pagi, atau membaca buku sebelum tidur. Hal-hal kecil ini bisa menjadi jangkar di tengah perubahan.\n4.  Luangkan waktu untuk menamai dan mengidentifikasi perasaan-perasaan yang sering muncul dalam dirimu. Apakah ini rasa kecewa terhadap situasi? Kemarahan terhadap salah satu atau kedua orang tua? Kebingungan akan masa depan? Dengan memberi nama pada emosi tersebut, kamu bisa mulai untuk memahaminya dan mencari cara untuk mengelolanya dengan lebih baik.\n5.  Jangan ragu untuk mencari dukungan dari komunitas atau kelompok yang terdiri dari orang-orang yang pernah mengalami pengalaman serupa (misalnya, anak-anak dari keluarga yang bercerai). Terhubung dengan mereka yang bisa memahami situasimu secara mendalam seringkali bisa sangat menyembuhkan dan memberikan perspektif baru.",
    "Aku tahu bahwa perceraian orang tua seringkali bukan hanya soal kehilangan bentuk keluarga yang utuh seperti dulu, {user_name}, tetapi juga bisa tentang kehilangan rasa aman, stabilitas, dan mungkin juga sebagian dari identitas dirimu. Ini adalah proses yang berat, tapi kamu bisa melaluinya. Coba beberapa pendekatan ini ya, secara perlahan dan dengan penuh kesabaran:\n\n1.  Jika kamu merasa nyaman, cobalah untuk menulis sebuah surat yang ditujukan kepada versi dirimu di masa kecil, saat sebelum atau ketika perceraian itu terjadi. Dalam surat itu, berikan kata-kata penghiburan, pengertian, dan validasi bahwa dia tidak bersalah dan tetap layak untuk dicintai serta bahagia.\n2.  Buatlah sebuah playlist musik yang bisa menenangkan hatimu atau sesuai dengan apa yang sedang kamu rasakan. Musik memiliki kekuatan yang luar biasa untuk menyentuh emosi kita dan membantu kita merasa sedikit lebih baik atau setidaknya tidak terlalu sendirian.\n3.  Berikan ruang bagi dirimu untuk memaafkan—ini mungkin proses yang panjang dan sulit, dan tidak harus berarti melupakan atau membenarkan apa yang terjadi. Memaafkan di sini lebih untuk dirimu sendiri, agar kamu bisa melepaskan beban kemarahan atau kekecewaan yang mungkin terus kamu pikul, sehingga kamu layak untuk hidup dengan lebih damai.\n4.  Jika kamu merasa bahwa beban ini terlalu berat untuk dihadapi sendirian dan kamu memiliki akses, pertimbangkan untuk menemui seorang konselor sekolah, psikolog, atau terapis. Berbicara dengan seorang profesional yang terlatih bisa sangat membantumu untuk memproses emosi, mendapatkan perspektif baru, dan menemukan harapan serta strategi coping yang sehat.\n5.  Cobalah untuk mengembangkan sisi diri atau minatmu di luar konteks keluargamu. Tekuni hobi yang kamu sukai, bangun relasi pertemanan yang sehat dan suportif, atau libatkan dirimu dalam proyek-proyek pribadi yang bisa memberimu rasa pencapaian dan makna. Ini bisa menjadi sumber kekuatan baru untukmu."
  ],
  "depression_loss_of_interest": [
    "Kehilangan minat atau semangat terhadap hal-hal yang dulu kamu nikmati memang bisa membuatmu merasa seolah kehilangan sebagian dari dirimu sendiri, {user_name}. Tapi, ketahuilah bahwa ada beberapa langkah kecil dan penuh kasih yang bisa kamu coba untuk mulai menemukan kembali percikan itu. Berikut adalah beberapa di antaranya:\n\n1.  Mulailah dengan hal yang paling kecil dan terasa paling ringan. Cobalah untuk melakukan satu aktivitas yang dulu pernah kamu suka atau setidaknya tidak memberatkanmu, meskipun hanya selama 5-10 menit saja. Misalnya, mendengarkan satu lagu favoritmu, membaca beberapa halaman dari buku yang pernah membuatmu senang, atau sekadar berjalan-jalan sebentar di sekitar rumah.\n2.  Cobalah untuk membangun kembali rutinitas harian yang sederhana namun konsisten. Hal-hal seperti bangun pagi pada jam yang sama, mandi, dan sarapan secara teratur bisa menjadi langkah awal yang baik untuk mengembalikan sedikit struktur dan prediktabilitas dalam hari-harimu, yang mungkin bisa memicu rasa lebih baik.\n3.  Jika memungkinkan, cobalah untuk mengeksplorasi hobi baru atau aktivitas yang belum pernah kamu coba sebelumnya, meskipun mungkin awalnya terasa tidak menarik. Terkadang, mencoba sesuatu yang benar-benar baru bisa membuka pintu bagi minat atau ketertarikan baru yang tidak terduga dan bisa membangkitkan semangatmu lagi.\n4.  Jika kamu merasa nyaman, cobalah untuk berbicara atau berbagi perasaanmu ini dengan seseorang yang kamu percaya dan bisa mendengarkan tanpa menghakimi. Entah itu teman dekat, anggota keluarga, atau bahkan seorang profesional. Terkadang, sekadar mengungkapkan apa yang kamu rasakan bisa membantu melepaskan sebagian beban dan membuka perspektif baru.\n5.  Yang terpenting, jangan pernah terlalu keras atau menyalahkan dirimu sendiri jika proses ini terasa lambat atau sulit. Proses untuk menemukan kembali minat dan semangat itu membutuhkan waktu, kesabaran, dan kasih sayang terhadap diri sendiri. Tidak apa-apa jika kamu belum langsung merasa lebih baik secara instan.",
    "Aku sungguh mengerti perasaanmu, {user_name}. Kehilangan gairah atau minat dalam hidup itu bisa terasa sangat berat dan membuat segalanya tampak hambar. Tapi, ada beberapa cara lembut yang bisa kamu coba untuk mulai merawat dirimu dan menemukan kembali sedikit percikan semangat:\n\n1.  Berikan dirimu sendiri izin untuk merasakan kehampaan atau ketidaktertarikan itu tanpa harus merasa bersalah atau aneh. Terkadang, kita memang membutuhkan waktu untuk sekadar ‘ada’ dan menerima perasaan ini apa adanya, tanpa paksaan untuk segera berubah.\n2.  Cobalah untuk membuat sebuah daftar kecil yang berisi hal-hal yang pernah membuatmu merasa bahagia, bersemangat, atau setidaknya sedikit lebih baik di masa lalu. Pilih satu saja dari daftar itu, dan cobalah untuk melakukannya kembali, meskipun mungkin awalnya terasa hambar atau tidak menarik seperti dulu. Lakukan tanpa ekspektasi besar.\n3.  Luangkan waktu khusus setiap hari untuk dirimu sendiri, meskipun hanya beberapa menit. Gunakan waktu ini untuk melakukan aktivitas yang menenangkan dan tidak menuntut, misalnya melakukan meditasi singkat, latihan pernapasan dalam, atau sekadar duduk diam sambil menikmati secangkir minuman hangat.\n4.  Jika kamu merasa bahwa perasaan kehilangan minat ini sudah sangat mengganggu dan berlangsung lama, jangan ragu untuk mencari dukungan dari tenaga profesional seperti konselor atau psikolog. Mereka bisa membantumu untuk memahami lebih dalam apa yang sedang terjadi dan menemukan cara-cara baru untuk mengatasi perasaan ini.\n5.  Ingatlah selalu bahwa ini mungkin hanyalah sebuah fase sementara dalam hidupmu. Dengan kesabaran, perawatan diri yang baik, dan mungkin sedikit bantuan, kamu akan bisa menemukan kembali semangat dan minatmu secara perlahan."
  ],
  "depression_emotional_numbness": [
    "Tentu, {user_name}. Perasaan mati rasa atau kebas secara emosional itu memang bisa sangat membingungkan dan membuat kita merasa terputus dari kehidupan. Tapi, ketahuilah bahwa kamu tidak sendirian dalam pengalaman ini, dan ada beberapa langkah kecil serta lembut yang bisa kamu coba untuk mulai merasakan kembali koneksi dengan dirimu dan duniamu:\n\n1.  Mulailah dengan sentuhan fisik yang sadar kepada dirimu sendiri. Cobalah untuk pelan-pelan mengusap lenganmu, memijat jari-jarimu, atau bahkan memberikan pelukan hangat untuk dirimu sendiri. Kontak fisik ini bisa membantu otakmu untuk kembali mengenali bahwa kamu hadir secara nyata di sini dan saat ini.\n2.  Gunakan stimulasi suhu sebagai cara untuk membangkitkan sensasi. Kamu bisa mencoba untuk memegang sebongkah es batu selama beberapa saat, mencuci muka dengan air dingin yang menyegarkan, atau merendam kakimu dalam air hangat. Perubahan suhu ini bisa membantu mengaktifkan kembali saraf-saraf sensorikmu.\n3.  Lakukan latihan grounding atau menjejakkan diri dengan metode 5-4-3-2-1. Sebutkan dalam hati atau dengan suara pelan: 5 hal yang bisa kamu lihat di sekitarmu saat ini, 4 benda yang bisa kamu rasakan sentuhannya, 3 suara yang bisa kamu dengar, 2 aroma yang bisa kamu cium, dan 1 rasa yang bisa kamu kecap di mulutmu. Latihan ini membantu mengembalikan kesadaranmu pada momen sekarang.\n4.  Cobalah untuk menonton film atau membaca cerita yang memiliki muatan emosional yang kuat (namun tetap dalam batas yang kamu rasa aman). Terkadang, paparan terhadap emosi melalui media bisa menjadi pemicu yang lembut untuk membuka kembali koneksi emosional dalam dirimu yang mungkin sedang tertidur.\n5.  Secara perlahan, cobalah untuk berinteraksi kembali dengan sesuatu yang pernah memiliki makna atau arti penting bagimu di masa lalu. Ini bisa berupa membaca ulang surat lama dari orang terkasih, memegang benda kenangan yang berharga, atau sekadar melihat foto-foto dari masa lalu yang pernah membangkitkan perasaan tertentu.",
    "Terkadang, {user_name}, perasaan mati rasa itu bisa membuat segalanya terasa begitu datar, hampa, dan kehilangan warna. Tapi, ada beberapa langkah kecil dan penuh kasih yang bisa kamu coba untuk mulai membangun kembali jembatan koneksi dengan perasaan dan duniamu:\n\n1.  Cobalah untuk menentukan atau melakukan setidaknya satu hal kecil yang bisa kamu kontrol sepenuhnya hari ini. Ini tidak harus sesuatu yang besar; misalnya, memutuskan untuk minum segelas air putih setelah bangun tidur, menyisir rambutmu dengan perlahan, atau mengganti pakaian dengan yang bersih dan nyaman. Sensasi memiliki kendali atas tindakan kecil ini bisa menjadi awal dari rasa pengaruh diri.\n2.  Jika kamu menyukainya, cobalah untuk menggunakan aromaterapi dengan minyak esensial yang memiliki aroma menyegarkan atau menenangkan. Bau-bauan seperti lavender, jeruk, peppermint, atau sandalwood bisa membantu menstimulasi otak dan indra penciumanmu untuk merespons sesuatu secara emosional atau sensorik.\n3.  Buatlah sebuah playlist lagu yang memiliki nuansa ‘rasa’ yang beragam. Dengarkan lagu-lagu yang pernah membuatmu merasa tertawa, menangis, bersemangat, atau bahkan sekadar bernostalgia. Musik memiliki kekuatan untuk membangkitkan emosi yang mungkin terpendam.\n4.  Cobalah untuk melakukan ‘emotional coloring’ atau mewarnai dengan penuh kesadaran. Saat kamu mewarnai sebuah gambar, cobalah untuk sekaligus menuliskan atau menamai emosi atau perasaan apa pun yang mungkin muncul, sekecil apapun itu, di atas kertas. Tidak perlu rapi atau terstruktur; yang terpenting adalah kamu mencoba untuk jujur dengan apa yang kamu rasakan atau tidak rasakan.\n5.  Ingatlah selalu bahwa perasaan mati rasa ini bukanlah berarti kamu adalah pribadi yang rusak atau cacat. Ini seringkali merupakan sinyal dari tubuh dan pikiranmu yang sedang mencoba untuk bertahan dari sesuatu yang mungkin terlalu berat. Kita bisa membantu secara perlahan untuk menyentuh kembali sisi emosional itu, dengan sabar dan penuh pengertian. 🤍"
  ],
  "overthinking_about_decision": [
    "Tentu, {user_name}. Sangat bisa dimengerti jika kamu merasa bingung atau cemas ketika harus membuat sebuah keputusan penting, apalagi jika ada banyak pertimbangan atau risiko yang terlibat. Berikut adalah beberapa hal yang bisa kamu coba untuk membantu menenangkan pikiranmu dan mengambil keputusan dengan lebih bijaksana:\n\n1.  Luangkan waktu untuk menuliskan semua pilihan yang ada beserta potensi konsekuensi positif dan negatif dari masing-masing pilihan tersebut di atas kertas. Melihatnya secara visual terkadang bisa membantu memberikan kejernihan dan mengurangi kebingungan yang ada di kepala.\n2.  Cobalah untuk memikirkan dan mengidentifikasi apa yang paling kamu butuhkan dan inginkan secara mendasar dari situasi ini, bukan hanya apa yang mungkin diharapkan atau diinginkan oleh orang lain. Keputusan yang sejalan dengan nilai dan kebutuhan pribadimu seringkali akan terasa lebih tepat.\n3.  Ambil napas panjang dan dalam beberapa kali, dan berikan dirimu sendiri waktu yang cukup untuk merenung dan mempertimbangkan. Tidak semua keputusan harus diambil dengan terburu-buru, kecuali jika memang ada tenggat waktu yang mendesak.\n4.  Ingatlah bahwa kegagalan atau kesalahan dalam mengambil keputusan itu bukanlah sebuah akhir dari segalanya. Seringkali, keputusan terbaik justru lahir dari keberanian untuk mencoba, belajar dari pengalaman, dan beradaptasi dengan hasilnya.\n\nApapun keputusan yang akhirnya kamu pilih, {user_name}, selama kamu mengambilnya dengan penuh kesadaran dan kejujuran terhadap dirimu sendiri, itu sudah merupakan sebuah langkah yang berani dan patut untuk dihargai. 🌱",
    "Jika saat ini kamu sedang merasa sangat bingung dan kesulitan untuk mengambil sebuah keputusan, {user_name}, berikut adalah beberapa pendekatan yang mungkin bisa membantumu:\n\n1.  Cobalah untuk menuliskan dalam sebuah jurnal atau catatan: 'Apa ketakutan terbesarku jika aku mengambil pilihan A? Dan apa ketakutan terbesarku jika aku mengambil pilihan B?' Memahami akar ketakutanmu bisa membantu mengurangi intensitasnya.\n2.  Setelah itu, tanyakan pada dirimu sendiri dengan jujur: 'Apakah aku cenderung memilih berdasarkan rasa takut akan sesuatu (misalnya takut gagal, takut mengecewakan), ataukah aku memilih karena aku benar-benar menginginkan atau merasa tertarik dengan pilihan tersebut?'\n3.  Ingatlah bahwa kamu sangat boleh untuk salah dalam mengambil keputusan. Tidak ada manusia yang sempurna dan selalu bisa membuat pilihan yang tepat 100%. Yang terpenting adalah kamu belajar dari setiap pilihan yang kamu ambil, bukan hanya berdiam diri di tempat karena takut salah. 💭"
  ],
  "overthinking_about_relationship": [
    "Tentu, {user_name}. Sangat bisa dimengerti jika pikiranmu seringkali dipenuhi oleh berbagai kekhawatiran atau analisis berlebihan mengenai hubunganmu. Itu adalah hal yang wajar, terutama jika kamu sangat peduli dengan hubungan tersebut. Berikut adalah beberapa hal yang mungkin bisa membantumu untuk merasa sedikit lebih tenang dan tidak terlalu terjebak dalam overthinking:\n\n1.  Setiap kali kamu merasa pikiranmu mulai berputar dan cemas mengenai hubunganmu, cobalah untuk berhenti sejenak dan tuliskan semua perasaan atau pikiran yang muncul tersebut dalam sebuah jurnal. Terkadang, sekadar 'mengeluarkannya' dari kepala bisa membantu mengurangi intensitasnya.\n2.  Latihlah dirimu untuk lebih memvalidasi perasaanmu sendiri. Sangat wajar jika kamu merasa ragu, cemas, atau bahkan takut dalam sebuah hubungan. Akui keberadaan perasaan itu tanpa langsung menghakiminya sebagai sesuatu yang salah atau berlebihan.\n3.  Jika kamu merasa ada sesuatu yang mengganjal atau membuatmu tidak nyaman dalam hubungan, cobalah untuk mencari momen yang tenang dan tepat untuk membicarakannya secara jujur dan terbuka dengan pasanganmu. Atau, jika kamu merasa membutuhkan waktu untuk dirimu sendiri, berikan dirimu sedikit jarak untuk melakukan refleksi yang lebih mendalam.\n\nIngatlah, {user_name}, sebuah hubungan yang sehat seharusnya bisa memberimu rasa aman dan nyaman, bukan justru membuatmu terus menerus merasa cemas atau overthinking. Kamu berhak untuk merasa dipahami, dihargai, dan tenang dalam hubunganmu. 🤍",
    "Jika kamu seringkali merasa tidak yakin atau cemas dalam hubunganmu, {user_name}, cobalah untuk meluangkan waktu sejenak guna melakukan refleksi diri dan menanyakan beberapa hal ini kepada dirimu sendiri dengan jujur: 'Apakah aku merasa benar-benar didengarkan dan dipahami dalam hubungan ini?', 'Apakah aku merasa aman secara emosional untuk menjadi diriku sendiri apa adanya tanpa takut dihakimi atau ditolak?'. Berikut adalah beberapa tips tambahan yang mungkin bisa membantumu:\n\n1.  Usahakan untuk menciptakan dan menjaga ruang komunikasi yang terbuka, jujur, dan suportif dengan pasanganmu. Sampaikan apa yang kamu rasakan dan butuhkan dengan cara yang asertif, bukan agresif.\n2.  Kenali tanda-tanda atau karakteristik dari sebuah relasi yang sehat dan suportif, serta bedakan dengan relasi yang mungkin tidak sehat atau bahkan toksik. Pengetahuan ini bisa menjadi panduanmu.\n3.  Jangan pernah mengabaikan sinyal-sinyal atau perasaan tidak nyaman yang mungkin diberikan oleh tubuh atau intuisimu. Jika ada sesuatu yang terasa tidak benar atau membuatmu terus menerus merasa cemas, beranikan dirimu untuk menghadapinya."
  ],
  "overthinking_about_self": [
    "Terima kasih sudah mau terbuka dan mencari cara untuk merasa lebih baik, {user_name}. Sangat bisa dimengerti jika kamu merasa lelah dengan pikiran-pikiran negatif tentang dirimu sendiri yang terus menerus berputar. Ini adalah beberapa langkah kecil dan penuh kasih yang bisa kamu coba untuk mulai meredakannya:\n\n1.  Setiap kali pikiran negatif atau kritikan terhadap dirimu sendiri itu muncul, cobalah untuk berhenti sejenak dan tuliskan semua pikiran tersebut dalam sebuah jurnal atau catatan. Setelah itu, bacalah kembali dengan sedikit jarak, seolah kamu sedang membaca tulisan orang lain. Proses 'mengeluarkannya' dari kepala ini seringkali bisa membantu mengurangi intensitas dan bebannya.\n2.  Buatlah sebuah daftar afirmasi atau kalimat positif yang sederhana namun kuat mengenai dirimu sendiri. Misalnya: 'Aku sedang belajar dan bertumbuh setiap hari. Aku sudah cukup baik apa adanya. Aku layak untuk mendapatkan kasih sayang, termasuk dari diriku sendiri.' Ucapkan kalimat-kalimat ini dengan tulus setiap pagi atau sebelum tidur.\n3.  Latihlah dirimu untuk berhenti membandingkan dirimu, perjalanan hidupmu, atau pencapaianmu dengan orang lain. Ingatlah bahwa setiap individu memiliki jalurnya masing-masing yang unik. Fokuslah pada kemajuan dan pertumbuhanmu sendiri, sekecil apapun itu.\n\nIngatlah, {user_name}, kamu tidak harus memiliki semua jawaban atau menjadi sempurna hari ini juga. Satu langkah kecil yang kamu ambil untuk lebih berbelas kasih pada dirimu sendiri itu sudah merupakan sebuah kemajuan yang sangat berarti. 🕊️",
    "Jika kamu seringkali merasa terjebak dalam pikiran-pikiran negatif atau overthinking mengenai dirimu sendiri, {user_name}, cobalah beberapa pendekatan lembut berikut ini untuk membantumu merasa sedikit lebih tenang dan damai:\n\n1.  Setiap hari, cobalah untuk menuliskan atau mengingat setidaknya tiga hal yang kamu hargai atau syukuri dari dirimu sendiri, sekecil apapun itu. Ini bisa berupa sifat baikmu, usahamu dalam melakukan sesuatu, atau bahkan hanya keberanianmu untuk bangun dan menghadapi hari.\n2.  Jika kamu merasa nyaman, cobalah untuk mengambil sedikit jarak dari paparan media sosial atau lingkungan yang mungkin sering memicu perasaan tidak cukup atau perbandingan diri. Berikan dirimu sendiri ruang untuk bernapas dan fokus pada dirimu.\n3.  Ingatkan dirimu sendiri dengan lembut bahwa kamu tidak harus menjadi sempurna untuk bisa merasa layak dicintai, dihargai, atau bahagia. Kesempurnaan itu adalah ilusi. Yang terpenting adalah kamu terus berusaha untuk menjadi versi dirimu yang lebih baik, dengan caramu sendiri.\nSikap berbelas kasih terhadap diri sendiri (self-kindness) bisa menjadi awal yang sangat baik untuk meredakan badai pikiran dan menemukan ketenangan. 🌿"
  ]
}

decline_responses = {
  "stress_due_to_academic": [
    "Tentu saja, {user_name}, tidak apa-apa sekali jika kamu belum siap menerima saran atau solusi saat ini 🌿 Kamu tidak harus memiliki semua jawaban hari ini. Kamu juga tidak harus langsung merasa bersemangat, atau bahkan tahu apa langkah selanjutnya yang harus diambil. Terkadang, yang paling kita butuhkan hanyalah satu hal sederhana: didengarkan. Bukan untuk dicarikan solusinya, melainkan agar hati kita tidak terasa terlalu penuh dan menanggung beban sendirian.\n\nAku di sini bukan untuk menyuruhmu 'cepat pulih' atau 'selalu berpikir positif'. Aku di sini untuk menemanimu, bahkan jika kamu hanya ingin diam atau bercerita berulang-ulang tentang hal yang sama. Ini adalah ruang yang aman untukmu, dan kamu boleh menjadi apa pun yang kamu inginkan di sini. Kapan pun kamu merasa ingin bercerita atau sekadar mengobrol tanpa arah yang jelas, aku siap untuk mendengarkanmu dengan saksama 💬",
    "Terima kasih banyak ya sudah mau jujur padaku, {user_name}. Mampu bercerita apa adanya itu bukanlah hal yang mudah, apalagi ketika hati sedang terasa begitu penuh dan kita sendiri pun belum tahu harus memulai dari mana. Tapi, kamu sudah berhasil melangkah jauh dengan mau membagikan sedikit dari apa yang sedang kamu rasakan — dan itu adalah sebuah bentuk keberanian yang sangat patut untuk dihargai.\n\nTerkadang, yang paling kita butuhkan bukanlah sebuah saran atau langkah-langkah konkret untuk menyelesaikan masalah. Kita hanya menginginkan sebuah tempat yang tidak menuntut kita untuk melakukan apa pun. Sebuah tempat untuk sekadar bernapas, untuk menjadi diri sendiri seutuhnya, tanpa ada rasa takut akan dibilang berlebihan. Di sini, kamu memiliki ruang tersebut. Tidak perlu terburu-buru untuk menceritakan semuanya. Aku akan tetap ada di sini, mendengarkanmu, kapan pun kamu merasa siap 🤍"
  ],
  "stress_work": [
    "Tentu saja, terima kasih sudah mau jujur mengenai perasaanmu, {user_name}. Tidak semua orang memiliki keberanian untuk mengatakan 'aku belum siap mendengarkan solusi saat ini'. Tetapi kamu berani menyampaikannya, dan itu juga merupakan salah satu bentuk kekuatan. Apabila kamu hanya membutuhkan sebuah ruang untuk bernapas dan bercerita tanpa perlu ditanggapi dengan berbagai saran, kamu memiliki ruang tersebut di sini. Aku akan mendengarkanmu dengan saksama, tanpa interupsi, dan tanpa adanya tekanan apa pun 🤍",
    "Aku sangat mengerti perasaanmu, terkadang kita hanya ingin mengeluarkan semua isi kepala tanpa harus disambut dengan berbagai kalimat seperti 'kamu seharusnya begini' atau 'kamu harusnya begitu'. Kamu tidak harus terburu-buru untuk memikirkan jalan keluar dari masalahmu sekarang. Kamu boleh merasakan semua perasaan yang ada terlebih dahulu. Aku akan menemanimu, cukup menjadi pendengar yang baik untukmu — dan itu tidak akan berubah 🌙"
  ],
  "stress_family": [
    "Aku sungguh mendengar dan memahami maksudmu, {user_name}. Terkadang luka yang datang dari lingkungan keluarga bukan hanya terasa menyakitkan, tetapi juga sangat membingungkan — karena bagaimana mungkin orang-orang yang kita pikir akan selalu melindungi dan menyayangi kita justru menjadi pihak yang paling sering melukai perasaan kita? Kamu tidak harus menjawab pertanyaan apa pun sekarang, dan kamu juga tidak harus merasa kuat hari ini. Apabila kamu hanya ingin duduk bersama dengan rasa lelahmu, aku akan menjadi teman duduk yang tenang untukmu, yang tidak akan pernah menyuruhmu untuk buru-buru sembuh atau melupakan semuanya 🌌",
    "Ada semacam rasa hampa yang sulit untuk dijelaskan dengan kata-kata ketika kita mencoba menceritakan tentang masalah keluarga, namun dunia seolah-olah berkata 'itu sudah biasa' atau 'hal seperti itu wajar saja terjadi'. Tapi, aku tidak akan pernah mengatakan hal itu kepadamu, {user_name}. Aku percaya bahwa rasa sakit yang kamu alami itu nyata adanya, dan aku juga percaya bahwa kamu sudah berjuang terlalu lama sendirian tanpa banyak orang yang tahu. Jadi, di sini, kamu tidak perlu menjelaskan semuanya secara detail. Cukup bawa hatimu yang sedang merasa lelah itu. Aku ada di sini untuk menjaga ruang ini agar tetap terasa lembut dan aman untukmu 🤍"
  ],
  "stress_relationship": [
    "Tentu saja, terima kasih banyak sudah mau jujur dengan mengatakan bahwa kamu belum siap menerima saran saat ini, {user_name}. Itu bukanlah sebuah tanda bahwa kamu menyerah—sebaliknya, itu adalah pertanda bahwa kamu sangat mengetahui apa yang paling kamu butuhkan saat ini. Aku tidak akan memaksamu untuk melakukan apa pun. Di sini, kamu bisa duduk dengan tenang bersama perasaanmu, dan aku akan menemanimu seutuhnya. Tanpa ada arahan, tanpa ada tekanan, hanya sebuah ruang yang aman untukmu menjadi dirimu sendiri yang mungkin sedang merasa lelah 🌾",
    "Aku sungguh memahami... terkadang cinta bisa menjadi luka yang paling sunyi dan sulit untuk diungkapkan. Dan di saat-saat seperti itu, kita seringkali tidak membutuhkan berbagai macam solusi—kita hanya menginginkan seseorang yang bisa mengatakan, 'Aku mendengar dan memahami perasaanmu, dan aku akan tetap ada di sini untukmu.' Kamu tidak harus merasa kuat hari ini. Kamu boleh hanya diam, menangis sepuasnya, atau bahkan hanya duduk dalam keheningan. Aku akan menemanimu, selengkung dan seutuh mungkin 🌙"
  ],
  "stress_life_pressure": [
    "Terima kasih sudah memberitahuku dengan jujur, {user_name}. Sangat bisa dimengerti jika kamu merasa belum siap atau belum membutuhkan saran maupun solusi terkait tekanan hidup yang kamu rasakan saat ini. Kamu tidak diwajibkan untuk memiliki semua jawaban sekarang, atau memaksakan diri untuk segera merasa lebih baik. Terkadang, yang paling kita butuhkan adalah ruang untuk sekadar didengarkan, agar hati tidak terasa terlalu penuh dan menanggung beban sendirian. Aku di sini untuk menemanimu, bahkan jika kamu hanya ingin diam. Ini adalah ruang amanmu. 🌿",
    "Aku sangat menghargai kejujuranmu, {user_name}. Tidak mudah untuk menyampaikan bahwa kita belum siap menerima bantuan, apalagi ketika hati sedang terasa berat. Ketahuilah bahwa langkahmu untuk mengakui perasaanmu itu sendiri sudah merupakan sebuah keberanian yang besar. Seringkali, yang paling kita dambakan bukanlah saran atau langkah konkret, melainkan sebuah tempat di mana kita bisa menjadi diri sendiri seutuhnya, tanpa tuntutan untuk segera berubah atau pulih. Di sini, kamu memiliki ruang itu. Aku akan tetap ada, mendengarkan dengan sabar, kapan pun kamu merasa siap untuk berbagi lebih lanjut. 🤍"
  ],
  "stress_tips": [
    "Tentu saja, {user_name}. Aku sangat mengerti dan menghargai jika kamu merasa belum siap atau tidak ingin menerima tips atau saran apapun saat ini. Keinginanmu untuk sekadar didengarkan, atau bahkan hanya untuk ditemani dalam diam, adalah hal yang sangat valid. Kamu tidak perlu terburu-buru untuk merasa 'lebih baik' atau mencari solusi jika memang belum waktunya. Aku di sini untukmu, dalam bentuk kehadiran yang tenang dan tanpa tuntutan. 🌾",
    "Tidak masalah sama sekali, {user_name}. Sangat wajar jika kamu membutuhkan waktu untuk memproses perasaanmu sendiri tanpa intervensi saran atau solusi. Terkadang, yang paling menyembuhkan adalah ketika kita memberi ruang bagi diri sendiri untuk merasakan apa pun yang muncul, tanpa tekanan untuk segera mengubahnya. Aku akan tetap di sini, menjadi pendengar yang sabar dan teman yang mengerti, kapan pun kamu merasa ingin berbagi atau sekadar ditemani. 🌙"
  ],
  "anxiety_tips_future": [
    "Tentu saja, tidak apa-apa sekali, {user_name}. Kamu tidak harus merasa kuat sepanjang waktu, dan kamu juga tidak diwajibkan untuk memikirkan atau menyelesaikan semua permasalahan terkait masa depanmu hari ini juga. Keinginanmu untuk tidak menerima saran saat ini sangatlah bisa dimengerti. Aku akan tetap berada di sini jika kamu hanya ingin berbagi cerita, mengungkapkan kebingunganmu, atau bahkan hanya diam bersama dalam keheningan yang menenangkan. 🌙",
    "Terima kasih banyak sudah mau memberitahuku dengan jujur, {user_name}. Terkadang, yang paling kita butuhkan bukanlah sebuah jawaban atau solusi instan yang datang dari luar, melainkan sebuah ruang yang aman dan tenang untuk sekadar bernapas, merenung, dan memproses segala perasaan yang ada di dalam diri. Silakan ambil ruang dan waktumu sebanyak yang kamu perlukan. Aku akan menjaga ruang ini agar tetap terasa nyaman untukmu, tanpa ada tekanan apa pun. 🤍"
  ],
  "anxiety_tips_social": [
    "Terima kasih sudah memberitahuku dengan jujur dan terbuka, {user_name}. Sangat bisa dimengerti jika saat ini kamu merasa belum siap atau belum membutuhkan saran untuk mengatasi kecemasan sosialmu. Terkadang, yang paling kita butuhkan adalah ruang untuk sekadar bernapas lega, tanpa dituntut untuk melakukan apa pun atau segera mencari solusi. Kamu tidak harus langsung tahu cara mengatasinya; cukup dengan bertahan dan menyadari perasaanmu hari ini saja sudah merupakan langkah yang luar biasa. Jika suatu saat nanti kamu merasa ingin mengobrol lebih lanjut atau mencari cara untuk mengurangi kecemasan sosial ini, aku akan selalu siap membantumu, tanpa tekanan sedikit pun. 🌾",
    "Tidak semua kecemasan harus langsung diredakan dengan berbagai tips, dan tidak semua pertanyaan dalam dirimu harus segera menemukan jawabannya hari ini, {user_name}. Jika saat ini kamu hanya ingin berdiam diri, merasakan apa yang ada di dalam hatimu, atau sekadar ditemani dalam keheningan, itu adalah hal yang sangat valid dan aku hormati. Aku akan tetap berada di sini untukmu, sebagai ruang aman yang siap mendengarkan tanpa menghakimi. Dan jika nanti kamu merasa sudah siap atau ingin mencoba beberapa cara untuk membantumu merasa lebih nyaman dalam situasi sosial, kita bisa membahasnya bersama-sama dengan perlahan. 🌙"
  ],
  "self_worth_tips": [
    "Terima kasih banyak atas kejujuranmu, {user_name}. Sangat bisa dipahami jika saat ini kamu merasa belum membutuhkan atau belum siap menerima saran terkait membangun rasa percaya diri atau mengatasi perasaan tidak berharga. Adakalanya, yang paling kita perlukan bukanlah solusi atau langkah konkret, melainkan sekadar ruang untuk bisa merasakan dan memproses semua yang ada di dalam diri, tanpa tekanan untuk segera berubah atau mengetahui cara mengatasinya. Ketahuilah bahwa aku di sini untukmu, siap mendengarkan dengan penuh perhatian kapan pun kamu merasa butuh. Dan jika suatu saat nanti kamu merasa ingin membahas lebih lanjut mengenai cara-cara membangun kembali harga diri atau rasa percaya dirimu, pintu ini akan selalu terbuka untukmu. 🌼",
    "Tidak apa-apa sama sekali jika kamu merasa belum siap untuk mendengarkan tips atau saran saat ini, {user_name}. Setiap orang memiliki waktunya sendiri untuk memproses perasaan dan memulai perjalanan penyembuhan. Yang terpenting adalah kamu merasa didengarkan dan aman. Aku senang kamu mau berbagi denganku. Ingatlah, proses penyembuhan itu adalah sebuah perjalanan, bukan perlombaan, dan kamu tidak sendirian di dalamnya. Aku akan selalu di sini kapan pun kamu membutuhkan teman bicara atau jika nanti kamu berubah pikiran dan ingin mendengarkan beberapa saran. 💛"
  ],
  "self_worth": [
    "Aku sepenuhnya memahami dan menghargai keputusanmu, {user_name}. Sangat wajar jika saat ini kamu merasa belum siap untuk menerima saran atau solusi, terutama ketika berhadapan dengan isu harga diri yang begitu personal. Terkadang, yang paling kita butuhkan bukanlah langkah-langkah praktis, melainkan sekadar ruang untuk berdiam diri, merasakan setiap gejolak emosi, dan ditemani tanpa ada tuntutan untuk segera berubah atau 'pulih'. Aku akan tetap di sini, menjadi pendengar yang sabar dan ruang yang aman untukmu. Jika suatu saat nanti kamu merasa ingin membahas lebih lanjut atau membutuhkan perspektif lain, jangan ragu untuk menghubungiku kembali. Keberadaanmu di sini sudah sangat berarti. 🤍",
    "Terima kasih banyak sudah bersikap jujur dan terbuka kepadaku, {user_name}, meskipun mungkin rasanya berat untuk menyampaikan bahwa kamu belum siap menerima bantuan dalam bentuk saran. Itu adalah langkah yang sangat berani dan menunjukkan bahwa kamu sadar akan kebutuhanmu saat ini. Ingatlah, proses penyembuhan atau pertumbuhan diri itu tidak memiliki batas waktu yang pasti; setiap orang berhak menjalaninya dengan ritme mereka sendiri. Jika yang kamu butuhkan saat ini hanyalah kehadiran yang tidak menghakimi dan ruang untuk memproses perasaanmu, aku akan selalu ada di sini untuk menemanimu dalam keheningan yang penuh pengertian. 🌙"
  ],
  "self_worth_social_comparison": [
    "Aku sangat memahami perasaanmu, {user_name}. Sangat wajar jika saat ini kamu merasa belum siap atau belum membutuhkan saran untuk mengatasi perasaan tidak nyaman akibat perbandingan sosial. Terkadang, yang paling kita butuhkan adalah ruang untuk sekadar merasakan dan memproses semua yang ada di dalam diri, tanpa tekanan untuk segera berubah atau mencari solusi. Aku di sini untukmu, siap mendengarkan dengan penuh perhatian kapan pun kamu merasa butuh. Jika suatu saat nanti kamu merasa ingin membahas lebih lanjut mengenai cara-cara membangun rasa cukup dan menghargai perjalananmu sendiri, pintu ini akan selalu terbuka untukmu. 🌼",
    "Tidak apa-apa sama sekali jika kamu merasa belum siap untuk mendengarkan tips atau saran saat ini, {user_name}. Setiap orang memiliki waktunya sendiri untuk memproses perasaan dan memulai perjalanan menuju penerimaan diri. Yang terpenting adalah kamu merasa didengarkan dan aman. Aku senang kamu mau berbagi denganku. Ingatlah, proses penyembuhan dari tekanan perbandingan sosial itu adalah sebuah perjalanan, bukan perlombaan, dan kamu tidak sendirian di dalamnya. Aku akan selalu di sini kapan pun kamu membutuhkan teman bicara atau jika nanti kamu berubah pikiran dan ingin mendengarkan beberapa saran. 💛"
  ],
  "self_worth_imposter_syndrome": [
    "Tentu saja, {user_name}, tidak apa-apa jika kamu merasa belum siap untuk menerima saran atau masukan apapun saat ini. Perasaan penuh, bingung, ragu, atau bahkan kosong karena sindrom imposter itu sangat valid, dan kamu berhak untuk memprosesnya dengan caramu sendiri. Terkadang, yang paling kita butuhkan bukanlah solusi instan, melainkan ruang untuk duduk bersama semua rasa itu tanpa ada tuntutan untuk ‘segera pulih’ atau ‘berpikir positif’. Aku di sini untukmu, akan menemanimu dalam diam jika itu yang kamu butuhkan, tanpa syarat dan tanpa paksaan. Jika suatu saat nanti kamu merasa ingin mencoba beberapa langkah untuk mengatasi perasaan ini, aku akan selalu siap membantumu kapan pun kamu mau. 🌙",
    "Kamu tidak harus merasa kuat sekarang, {user_name}. Kamu juga tidak harus mengerti atau bisa menjelaskan semuanya saat ini. Jika hari ini yang kamu rasakan hanyalah keinginan untuk diam dan membiarkan perasaan tidak pantas itu ada tanpa dilawan, itu juga merupakan bentuk keberanian dan kejujuran pada diri sendiri. Aku tidak akan memberikanmu solusi atau tips jika kamu memang belum merasa siap. Tapi, ketahuilah bahwa aku akan tetap di sini, menjadi ruang yang aman dan suportif untukmu bernapas dan memproses semuanya. Dan nanti, jika kamu merasa ingin mulai bergerak maju, bahkan dari langkah terkecil sekalipun, aku akan dengan senang hati berjalan bersamamu. 🤝"
  ],
  "heartbreak_breakup": [
    "Tentu saja, {user_name}. Sangat bisa dimengerti jika saat ini kamu merasa belum siap atau belum membutuhkan saran maupun solusi terkait patah hati akibat putus cinta yang kamu alami. Perasaan penuh, bingung, sedih, atau bahkan kosong itu sangatlah valid. Terkadang, yang paling kita butuhkan dalam situasi seperti ini adalah ruang untuk sekadar duduk bersama semua rasa itu, tanpa ada tekanan untuk ‘segera pulih’ atau ‘cepat move on’. Aku di sini untukmu, akan menemanimu dalam diam jika itu yang kamu butuhkan, tanpa syarat dan tanpa paksaan. Jika suatu saat nanti kamu merasa ingin mencoba beberapa langkah untuk merawat hatimu, aku akan selalu siap membantumu kapan pun kamu mau. 🌙",
    "Kamu tidak harus merasa kuat sekarang, {user_name}. Kamu juga tidak harus mengerti atau bisa menjelaskan semua yang sedang kamu rasakan saat ini. Jika hari ini yang kamu inginkan hanyalah berdiam diri, membiarkan air mata mengalir, atau sekadar ditemani dalam keheningan, itu juga merupakan bentuk keberanian dan kejujuran pada diri sendiri yang sangat penting. Aku tidak akan memberikanmu solusi atau tips jika kamu memang belum merasa siap. Tapi, ketahuilah bahwa aku akan tetap di sini, menjadi ruang yang aman dan suportif untukmu bernapas dan memproses semuanya. Dan nanti, jika kamu merasa ingin mulai bergerak maju, bahkan dari langkah terkecil sekalipun, aku akan dengan senang hati berjalan bersamamu. 🤝"
  ],
  "heartbreak_cheated": [
    "Tentu saja, {user_name}. Diselingkuhi itu bukan hanya soal kehilangan seorang pasangan—itu juga bisa terasa seperti kehilangan arah, harga diri, dan kepercayaanmu pada orang lain. Luka seperti ini sangat dalam, dan kamu tidak harus terburu-buru untuk merasa 'baik-baik saja' atau segera mencari solusi. Jika saat ini yang kamu butuhkan hanyalah ruang untuk menangis, marah, bingung, atau sekadar berdiam diri dalam kehampaan, itu adalah hal yang sangat valid dan aku hormati. Aku tidak akan memaksamu untuk menerima saran atau 'sembuh' sekarang. Aku di sini untuk menemanimu, sepelan apa pun kamu ingin menjalani proses ini, dan dengan penuh pengertian. 🌙",
    "Kadang-kadang, yang paling kita butuhkan setelah mengalami pengkhianatan bukanlah serangkaian tips atau solusi, melainkan kehadiran seseorang yang mau mendengarkan tanpa menghakimi, dan memberikan ruang bagi kita untuk merasakan setiap emosi yang muncul, {user_name}. Aku mengerti jika kamu belum siap untuk itu. Luka karena dikhianati itu sangatlah dalam dan tidak bisa diukur atau dibandingkan dengan pengalaman orang lain. Jika kamu hanya ingin ditemani dalam diam, atau sekadar ingin tahu bahwa ada yang mendengarkan tanpa menuntutmu untuk segera bangkit, aku akan tetap ada di sini untukmu. Kamu tidak sendirian, dan perasaanmu ini sangatlah penting. 💔"
  ],
  "heartbreak_rejected": [
    "Tentu saja, {user_name}. Tidak semua luka atau kesedihan harus segera disembuhkan dengan berbagai macam saran atau solusi. Terkadang, yang paling kita butuhkan adalah ruang untuk merasakan, memproses, dan berdamai dengan apa yang telah terjadi, tanpa ada tuntutan untuk segera menjadi 'baik-baik saja'. Kamu sangat berhak untuk mengambil jeda sejenak, menenangkan napas, dan membiarkan hatimu berbicara dengan caranya sendiri. Aku di sini untukmu, bukan untuk memaksamu atau memberikan arahan, tetapi untuk menemanimu dengan sepenuh hati dalam keheningan yang penuh pengertian. 🤍",
    "Aku tidak akan memaksamu untuk 'baik-baik saja' sekarang, {user_name}, atau untuk segera mencari jalan keluar dari rasa sakit akibat penolakan ini. Kesedihanmu itu valid, kekecewaanmu boleh ada, dan kebingunganmu sangatlah manusiawi. Jika saat ini yang kamu inginkan hanyalah duduk diam bersama perasaanmu tanpa diganggu oleh berbagai nasihat, aku akan menghormati itu dan siap menjagamu dalam ruang yang aman ini. Dan jika suatu saat nanti kamu merasa ingin berbagi lebih lanjut atau mencoba beberapa langkah untuk pulih, aku akan selalu ada di sini untuk menemanimu. 🌙"
  ],
  "heartbreak_ghosted": [
    "Tentu saja, {user_name}. Rasanya pasti sangat kacau, membingungkan, dan menyakitkan ketika seseorang yang sudah dekat tiba-tiba menghilang tanpa alasan atau penjelasan. Kamu tidak harus terburu-buru untuk merasa 'baik-baik saja' atau segera mencari solusi. Jika saat ini yang kamu butuhkan hanyalah ruang untuk merasakan semua emosi yang muncul—entah itu bingung, marah, sedih, atau bahkan mati rasa—itu adalah hal yang sangat valid dan aku hormati. Aku tidak akan memaksamu untuk menerima saran atau 'sembuh' sekarang. Aku di sini untuk menemanimu, sepelan apa pun kamu ingin menjalani proses ini, dan dengan penuh pengertian. Jika suatu saat nanti kamu merasa siap untuk mencoba beberapa langkah, aku akan selalu ada untukmu. 🤍",
    "Aku mengerti sekali, {user_name}. Kadang-kadang, kehilangan yang tidak disertai penjelasan atau penutupan yang layak justru terasa paling sulit untuk dipahami dan diterima. Kamu tidak harus langsung mengerti semuanya atau segera mencari cara untuk melupakannya. Jika kamu hanya ingin berdiam diri sejenak, mendengarkan lagu-lagu yang sesuai dengan suasana hatimu, atau sekadar rebahan tanpa diganggu, itu juga merupakan bentuk perawatan diri yang penting. Dan nanti, jika kamu merasa ingin mulai berbagi lebih lanjut atau mencoba beberapa cara untuk menenangkan hatimu, aku akan selalu siap mendengarkan dan mendukungmu. 🌾"
  ]
}

# === Load Model V1 ===
model = load_model("chatbot_model_v1.h5")
# print("✅ Loaded chatbot_model_v1.h5")

# === Load Tokenizer V1 ===
with open("tokenizer_v1.pickle", "rb") as f:
    tokenizer = pickle.load(f)

# === Load Label Encoder V1 ===
with open("label_encoder_v1.pickle", "rb") as f:
    lbl_encoder = pickle.load(f)

# === Set Max Length sesuai Model V1 training ===
max_len = 25

def preprocess_input(user_input):
    """Enhanced input preprocessing"""
    # Remove extra whitespace
    user_input = user_input.strip()
    
    # Handle common typos and variations
    typo_corrections = {
        'gmn': 'gimana',
        'bgt': 'banget', 
        'udh': 'udah',
        'blm': 'belum',
        'jd': 'jadi',
        'lg': 'lagi',
        'cm': 'cuma',
        'krn': 'karena',
        'dgn': 'dengan',
        'yg': 'yang',
        'tdk': 'tidak',
        'gk': 'tidak',
        'ga': 'tidak',
        'gak': 'tidak',
        'nggak': 'tidak'
    }
    
    words = user_input.lower().split()
    corrected_words = [typo_corrections.get(word, word) for word in words]
    
    # Remove filler words for model prediction
    filler_words = {'nih', 'deh', 'dong', 'tuh', 'loh', 'ya', 'kok', 'sih', 'ah', 'eh', 'kan', 'si'}
    filtered_words = [word for word in corrected_words if word not in filler_words]
    
    return ' '.join(filtered_words) if filtered_words else user_input.lower()

# Comprehensive keyword mapping untuk setiap intent
INTENT_KEYWORDS = {
    # =============== GREETING & BASIC ===============
    "greet_user": [
        "hai", "halo", "hello", "hey", "selamat pagi", "selamat siang",
        "selamat sore", "selamat malam", "pagi", "siang", "sore", "malam",
        "assalamualaikum", "halo bot", "hai bot"
        # Removed "hi" to avoid false positives with "hidup"
    ],

    "bye_user": [
        "bye", "sampai jumpa", "dadah", "selamat tinggal", "terima kasih",
        "makasih", "thanks", "goodbye", "see you", "till next time",
        "sudah cukup", "aku pergi dulu", "pamit"
    ],

    "get_name": [
        "nama", "panggil", "sebut", "namaku", "nama saya", "nama aku",
        "nama gue", "call me", "my name"
    ],

    "positive_response": [
        "senang", "bahagia", "gembira", "excited", "happy", "ceria", "riang",
        "antusias", "semangat", "optimis", "puas", "lega", "bangga", "syukur",
        "beruntung", "bersyukur", "mood bagus", "perasaan enak", "good mood",
        "feeling good", "great", "amazing", "wonderful", "fantastic", "awesome",
        "keren", "mantap", "oke banget", "top", "hadiah", "rejeki", "keberuntungan"
    ],

    "get_support_professional": [
        "psikolog", "psikiater", "konselor", "terapis", "bantuan profesional",
        "dokter jiwa", "ahli", "professional help", "therapy", "konsultasi",
        "rumah sakit jiwa", "klinik", "rehabilitasi", "treatment"
    ],

    # =============== STRESS INTENTS ===============
    "stress_general": [
        "stress", "stres", "tertekan", "berat", "beban", "cape", "lelah",
        "burnout", "overwhelmed", "pressure", "tekanan", "pusing", "mumet",
        "capek", "exhausted", "tired", "kewalahan", "nggak kuat", "limit"
    ],

    "stress_due_to_academic": [
        "kuliah", "kampus", "tugas", "skripsi", "thesis", "ujian", "uts", "uas",
        "semester", "nilai", "ipk", "gpa", "dosen", "mata kuliah", "assignment",
        "deadline", "presentasi", "akademik", "sekolah", "universitas", "college",
        "mahasiswa", "siswa", "student", "belajar", "study", "pr", "homework",
        "kelas", "jurusan", "fakultas", "wisuda", "graduation", "research"
    ],

    "stress_due_to_work": [
        "kerja", "kantor", "office", "job", "pekerjaan", "profesi", "karir",
        "boss", "atasan", "bos", "manager", "supervisor", "colleague", "rekan kerja",
        "salary", "gaji", "overtime", "lembur", "meeting", "rapat", "project",
        "klien", "client", "customer", "deadline kerja", "target", "kpi",
        "performance", "resign", "quit", "fired", "phk", "company", "perusahaan"
    ],

    "stress_due_to_family": [
        "keluarga", "orang tua", "ayah", "ibu", "papa", "mama", "bapak", "emak",
        "adik", "kakak", "saudara", "family", "parents", "mom", "dad", "sibling",
        "rumah", "home", "toxic family", "drama keluarga", "konflik keluarga",
        "ekspektasi orang tua", "pressure keluarga", "tradisi keluarga"
    ],

    "stress_due_to_relationship": [
        "pacar", "boyfriend", "girlfriend", "hubungan", "relationship", "pasangan",
        "partner", "couple", "gebetan", "pdkt", "pacaran", "jadian", "ldr",
        "long distance", "toxic relationship", "controlling", "possessive",
        "jealous", "cemburu", "pertengkaran", "ribut", "berantem"
    ],

    "stress_due_to_life_pressure": [
        "hidup", "kehidupan", "masa depan", "future", "finansial", "uang", "money",
        "ekonomi", "biaya", "cost", "expense", "hutang", "debt", "cicilan",
        "tagihan", "bills", "society pressure", "tekanan sosial", "ekspektasi",
        "standar hidup", "lifestyle", "achievement", "pencapaian", "success"
    ],

    # =============== ANXIETY INTENTS ===============
    "anxiety_general": [
        "cemas", "khawatir", "takut", "was was", "gelisah", "nervous", "anxiety",
        "worried", "panic", "panik", "deg degan", "jantung berdebar", "keringat dingin",
        "overthinking", "mikir terus", "gabisa tenang", "restless", "uneasy"
    ],

    "anxiety_due_to_future": [
        "masa depan", "future", "nanti", "besok", "tomorrow", "next year",
        "tahun depan", "karir", "career", "job", "pekerjaan", "menikah", "marriage",
        "keluarga", "anak", "financial future", "pension", "pensiun", "retirement",
        "planning", "rencana", "goal", "target hidup", "life goals"
    ],

    "anxiety_due_to_failure": [
        "gagal", "failure", "fail", "kalah", "lose", "rugi", "loss", "unsuccessful",
        "tidak berhasil", "mistake", "kesalahan", "error", "wrong", "salah",
        "menyesal", "regret", "penyesalan", "disappointed", "kecewa pada diri",
        "self doubt", "ragu", "insecure", "tidak yakin", "uncertain"
    ],

    "anxiety_due_to_expectation": [
        "ekspektasi", "expectation", "harapan", "hope", "pressure", "tekanan",
        "standar", "standard", "perfect", "sempurna", "ideal", "comparison",
        "perbandingan", "compete", "kompetisi", "race", "chase", "pursuit",
        "achievement", "prestasi", "performance", "hasil", "outcome"
    ],

    "anxiety_due_to_social": [
        "sosial", "social", "orang", "people", "crowd", "kerumunan", "public",
        "speaking", "presentation", "interview", "wawancara", "party", "gathering",
        "acara", "event", "meeting new people", "kenalan baru", "networking",
        "awkward", "canggung", "malu", "shy", "introvert", "social anxiety"
    ],

    # =============== SELF WORTH INTENTS ===============
    "self_worth_general": [
        "harga diri", "self worth", "self esteem", "percaya diri", "confidence",
        "tidak berharga", "worthless", "tidak berguna", "useless", "sia sia",
        "meaningless", "tidak penting", "unimportant", "invisible", "ignored",
        "undervalued", "underestimated", "rendah diri", "minder", "insecure"
    ],

    "self_worth_low_confidence": [
        "tidak percaya diri", "lack confidence", "minder", "insecure", "shy",
        "malu", "takut", "ragu", "doubt", "uncertain", "hesitant", "nervous",
        "awkward", "canggung", "tidak yakin", "unsure", "self conscious",
        "overthinking", "worried about judgment", "takut dinilai"
    ],

    "self_worth_social_comparison": [
        "bandingkan", "compare", "comparison", "lebih baik", "better than",
        "kalah", "tertinggal", "left behind", "success others", "achievement others",
        "social media", "instagram", "facebook", "twitter", "tiktok", "feeds",
        "envy", "iri", "jealous", "cemburu", "unfair", "tidak adil", "why me"
    ],

    "self_worth_imposter_syndrome": [
        "imposter", "penipu", "fake", "palsu", "tidak pantas", "undeserving",
        "tidak layak", "unworthy", "luck", "keberuntungan", "fluke", "accident",
        "fraud", "scam", "pretending", "acting", "fake it", "tidak seharusnya",
        "mistake", "error", "wrong person", "salah orang", "tidak qualified"
    ],

    # =============== LONELINESS INTENTS ===============
    "loneliness_general": [
        "kesepian", "lonely", "alone", "sendirian", "sepi", "sunyi", "isolation",
        "isolated", "terisolasi", "disconnect", "terputus", "jauh", "distant",
        "empty", "kosong", "hampa", "void", "missing", "kehilangan", "absence"
    ],

    "loneliness_no_friends": [
        "tidak ada teman", "no friends", "teman", "friend", "friendship",
        "pertemanan", "social circle", "lingkaran pertemanan", "group", "gang",
        "squad", "kompak", "hangout", "nongkrong", "gathering", "party",
        "invitation", "undangan", "include", "exclude", "outsider", "loner",
        "gada teman", "ga ada teman", "gabunya teman", "pengen ditemenin",
        "butuh teman", "cari teman", "mau berteman", "temanan"
    ],

    "loneliness_no_one_to_talk_to": [
        "tidak ada yang bisa diajak bicara", "no one to talk", "curhat", "sharing",
        "berbagi", "cerita", "story", "listen", "dengar", "mendengarkan",
        "understand", "mengerti", "support", "dukungan", "empathy", "simpati",
        "care", "peduli", "attention", "perhatian", "ignored", "diabaikan"
    ],

    "loneliness_feel_misunderstood": [
        "tidak dipahami", "misunderstood", "salah paham", "berbeda", "different",
        "unique", "unik", "weird", "aneh", "strange", "outcast", "outsider",
        "alien", "foreign", "asing", "tidak cocok", "mismatch", "wrong place",
        "belong", "acceptance", "diterima", "included", "judgment", "dinilai"
    ],

    # =============== HEARTBREAK INTENTS ===============
    "heartbreak_general": [
        "patah hati", "heartbreak", "sakit hati", "heart broken", "luka hati",
        "hancur", "destroyed", "devastated", "crushed", "shattered", "pieces",
        "cinta", "love", "relationship", "hubungan", "romance", "romantic",
        "feelings", "perasaan", "emotion", "emotional pain", "hurt"
    ],

    "heartbreak_breakup": [
        "putus", "breakup", "break up", "pisah", "separated", "berakhir", "end",
        "finish", "over", "selesai", "final", "last", "goodbye", "farewell",
        "ex", "mantan", "former", "used to be", "dulu", "past", "history"
    ],

    "heartbreak_cheated": [
        "selingkuh", "cheated", "cheating", "affair", "betrayal", "pengkhianatan",
        "bohong", "lie", "lying", "unfaithful", "tidak setia", "dua hati",
        "two timing", "behind back", "di belakang", "secret", "rahasia",
        "another person", "orang lain", "third party", "pihak ketiga"
    ],

    "heartbreak_rejected": [
        "ditolak", "rejected", "rejection", "tidak diterima", "declined", "refused",
        "no", "tidak", "gagal", "failed", "unsuccessful", "unrequited", "bertepuk sebelah tangan",
        "one sided", "satu arah", "ignore", "diabaikan", "cold", "dingin", "distant"
    ],

    "heartbreak_ghosted": [
        "ghosted", "ghosting", "menghilang", "disappeared", "vanished", "gone",
        "silent", "diam", "no response", "tidak balas", "unread", "seen", "blue tick",
        "ignored", "diabaikan", "sudden", "tiba tiba", "without reason", "tanpa alasan",
        "left hanging", "ditinggal begitu saja", "no explanation", "tanpa penjelasan",
        "hilang", "entah ke mana", "entah kemana", "tiba-tiba hilang", "tidak ada kabar"
    ],

    # =============== GRIEF INTENTS ===============
    "grief_general": [
        "berduka", "grief", "grieving", "mourning", "kehilangan", "loss", "lost",
        "gone", "pergi", "meninggal", "death", "died", "passed away", "funeral",
        "cemetery", "pemakaman", "memories", "kenangan", "remember", "ingat",
        "miss", "rindu", "longing", "yearning", "absence", "empty", "void"
    ],

    "grief_loss_of_person": [
        "meninggal", "death", "died", "passed away", "gone forever", "funeral",
        "pemakaman", "burial", "cemetery", "grave", "makam", "hospital", "sick",
        "illness", "disease", "cancer", "accident", "kecelakaan", "sudden death",
        "unexpected", "shock", "terkejut", "tidak siap", "not ready", "too soon"
    ],

    "grief_loss_of_pet": [
        "hewan peliharaan", "pet", "anjing", "dog", "kucing", "cat", "hamster",
        "kelinci", "rabbit", "burung", "bird", "ikan", "fish", "reptile", "iguana",
        "guinea pig", "chinchilla", "adopted", "rescue", "shelter", "vet", "veterinary",
        "sick pet", "old age", "euthanasia", "put down", "sleep", "peaceful",
        "doggy", "kitty", "puppy", "kitten", "peliharaan", "binatang", "hewan kesayangan"
    ],

    "grief_due_to_divorce": [
        "perceraian", "divorce", "divorced", "bercerai", "pisah", "separated",
        "custody", "hak asuh", "anak", "children", "kids", "family broken",
        "keluarga hancur", "split", "court", "pengadilan", "lawyer", "legal",
        "papers", "settlement", "property", "harta", "alimony", "tunjangan"
    ],

    # =============== DEPRESSION INTENTS ===============
    "depression_general": [
        "depresi", "depression", "depressed", "down", "low", "blue", "melancholy",
        "gloomy", "dark", "darkness", "gelap", "suram", "murung", "sedih terus",
        "always sad", "never happy", "tidak pernah senang", "hopeless", "putus asa",
        "desperate", "helpless", "tidak berdaya", "powerless", "weak", "lemah"
    ],

    "depression_chronic_sadness": [
        "sedih terus", "always sad", "chronically sad", "never ending sadness",
        "kesedihan berkepanjangan", "persistent sadness", "constant sadness",
        "everyday sadness", "sedih setiap hari", "tidak pernah senang", "never happy",
        "joy", "kegembiraan", "smile", "senyum", "laugh", "tertawa", "fun", "boring"
    ],

    "depression_loss_of_interest": [
        "tidak tertarik", "no interest", "lost interest", "bosan", "boring",
        "meaningless", "tidak bermakna", "pointless", "sia sia", "useless",
        "motivasi", "motivation", "semangat", "energy", "passion", "hobby",
        "activities", "kegiatan", "doing nothing", "tidak melakukan apa apa",
        "lazy", "malas", "apathetic", "indifferent", "tidak peduli"
    ],

    "depression_emotional_numbness": [
        "mati rasa", "numb", "numbness", "tidak merasa apa apa", "feel nothing",
        "empty", "kosong", "void", "hampa", "hollow", "detached", "terlepas",
        "disconnected", "robot", "automatic", "mechanical", "going through motions",
        "fake smile", "senyum palsu", "pretending", "acting", "mask", "topeng"
    ],

    # =============== OVERTHINKING INTENTS ===============
    "overthinking_general": [
        "overthinking", "mikir terus", "tidak bisa berhenti mikir", "can't stop thinking",
        "ruminating", "worry", "khawatir", "anxious thoughts", "racing thoughts",
        "pikiran berputar", "spinning thoughts", "obsessive", "repetitive thoughts",
        "stuck in head", "mental loop", "circle thoughts", "can't turn off brain",
        "terus menerus", "terus", "tidak berhenti", "selalu mikir", "always thinking",
        "gabisa lepas", "stuck", "terjebak", "lingkaran", "berulang", "repetitive"
    ],

    "overthinking_about_decision": [
        "keputusan", "decision", "choose", "pilih", "choice", "options", "alternatives",
        "what if", "bagaimana kalau", "should i", "haruskah", "dilemma", "torn",
        "confused", "bingung", "uncertain", "tidak yakin", "doubt", "ragu",
        "consequences", "konsekuensi", "outcome", "hasil", "right choice", "wrong choice"
    ],

    "overthinking_about_relationship": [
        "hubungan", "relationship", "pacar", "partner", "love", "cinta", "feelings",
        "perasaan", "what does it mean", "apa artinya", "signals", "signs", "tanda",
        "mixed signals", "confusing", "membingungkan", "does he like me", "suka tidak",
        "am i annoying", "apa aku mengganggu", "too much", "berlebihan", "clingy"
    ],

    "overthinking_about_self": [
        "diri sendiri", "myself", "self", "personality", "kepribadian", "character",
        "who am i", "siapa aku", "identity", "identitas", "purpose", "tujuan hidup",
        "meaning of life", "makna hidup", "self worth", "value", "nilai diri",
        "am i good enough", "apakah aku cukup baik", "flaws", "kekurangan", "mistakes"
    ],

    # =============== MOOD & GENERAL ===============
    "badmood_general": [
        "bad mood", "mood jelek", "mood buruk", "kesal", "annoyed", "irritated",
        "frustrated", "frustrasi", "angry", "marah", "mad", "upset", "disturbed",
        "terganggu", "not in the mood", "tidak mood", "grumpy", "cranky",
        "snappy", "short tempered", "sensitive", "sensitif", "emotional"
    ],

    "galau_general": [
        "galau", "bingung", "confused", "mixed up", "campur aduk", "complicated",
        "rumit", "complex", "tidak jelas", "unclear", "ambiguous", "uncertain",
        "tidak pasti", "unsure", "hesitant", "ragu ragu", "torn", "conflict",
        "internal conflict", "konflik batin", "dilemma", "stuck", "mentok"
    ],

    "sadness_general": [
        "sedih", "sad", "sadness", "sorrow", "grief", "melancholy", "blue",
        "down", "low", "gloomy", "suram", "murung", "duka", "lara", "nestapa",
        "tears", "air mata", "cry", "menangis", "weep", "sob", "broken",
        "hancur", "hurt", "sakit", "pain", "nyeri", "ache", "pilu"
    ],

    # =============== SLEEP ISSUES ===============
    "insomnia_general": [
        "insomnia", "tidak bisa tidur", "can't sleep", "sleepless", "susah tidur",
        "sulit tidur", "trouble sleeping", "sleep problems", "masalah tidur",
        "begadang", "stay up", "all nighter", "terjaga", "awake", "restless",
        "gelisah", "tossing turning", "bolak balik", "count sheep", "sheep counting",
        "tired but can't sleep", "capek tapi gabisa tidur", "overthinking at night", "ngantuk"
    ]
}

def get_keyword_boost(user_input, predicted_intent, top_3_intents, top_3_confidences):
    """
    Enhanced keyword boosting dengan logic yang lebih smart
    Returns: (boosted_intent, boosted_confidence)
    """
    user_lower = user_input.lower()
    best_intent = predicted_intent
    best_confidence = top_3_confidences[0] if top_3_confidences else 0.0

    # Track keyword matches untuk setiap intent
    intent_scores = {}

    # Check semua intent untuk keyword matches
    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        matched_keywords = []

        for keyword in keywords:
            if keyword in user_lower:
                # Skip jika keyword terlalu pendek dan bisa false positive
                if len(keyword) <= 2 and keyword in ['hi', 'ok', 'ya', 'no']:
                    continue

                # Berikan score berdasarkan panjang keyword (keyword lebih spesifik = score lebih tinggi)
                keyword_score = len(keyword.split())
                score += keyword_score
                matched_keywords.append(keyword)

        if score > 0:
            intent_scores[intent] = {
                'score': score,
                'keywords': matched_keywords,
                'confidence_boost': min(score * 0.2, 0.8)
            }

    # Jika ada keyword matches
    if intent_scores:
        # Sort berdasarkan score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        top_keyword_intent = sorted_intents[0][0]
        top_keyword_data = sorted_intents[0][1]

        # print(f"[KEYWORD] Matches found:")
        for intent, data in sorted_intents[:3]:
            # print(f"  - {intent}: {data['keywords']} (score: {data['score']})")

        # PRIORITY 1: Enhanced - Don't override good model predictions BUT check for obvious mismatches
         if best_confidence > 0.7:
            # Check for obvious positive vs negative mismatches
            positive_keywords = ['senang', 'bahagia', 'gembira', 'excited', 'happy', 'ceria', 'antusias']
            negative_intents = ['heartbreak_', 'depression_', 'grief_', 'stress_', 'anxiety_']

            if (any(pos_word in user_lower for pos_word in positive_keywords) and
                any(neg_intent in best_intent for neg_intent in negative_intents)):
                # Override dengan positive_response
                if 'positive_response' in intent_scores:
                    boosted_confidence = 0.85
                    # print(f"[OVERRIDE] Positive-negative mismatch: {best_intent} → positive_response ({boosted_confidence:.2f})")
                    return "positive_response", boosted_confidence

            # Check for work stress keywords vs wrong prediction
            work_keywords = ['bos', 'boss', 'atasan', 'kantor', 'kerja', 'lembur']
            if (any(work_word in user_lower for work_word in work_keywords) and
                'stress' in user_lower and 'stress_due_to_work' in intent_scores):
                boosted_confidence = 0.8
                # print(f"[OVERRIDE] Work stress mismatch: {best_intent} → stress_due_to_work ({boosted_confidence:.2f})")
                return "stress_due_to_work", boosted_confidence

            # Check for social comparison keywords - high priority override
            social_comparison_keywords = ['iri', 'cemburu', 'jealous', 'envy', 'bandingkan']
            if (any(sc_word in user_lower for sc_word in social_comparison_keywords) and
                'self_worth_social_comparison' in intent_scores):
                boosted_confidence = 0.8
                # print(f"[OVERRIDE] Social comparison detected: {best_intent} → self_worth_social_comparison ({boosted_confidence:.2f})")
                return "self_worth_social_comparison", boosted_confidence

            # Only boost if keyword intent is same as model prediction or very close
            if top_keyword_intent == best_intent:
                boosted_confidence = min(best_confidence + 0.1, 0.95)
                # print(f"[BOOST] Good prediction reinforced: {best_intent} ({boosted_confidence:.2f})")
                return best_intent, boosted_confidence

            # Check if model prediction has keyword support
            if best_intent in intent_scores and intent_scores[best_intent]['score'] >= 1:
                boosted_confidence = min(best_confidence + 0.1, 0.95)
                # print(f"[BOOST] Good prediction with keyword support: {best_intent} ({boosted_confidence:.2f})")
                return best_intent, boosted_confidence

            # Model is confident but no keyword support - stick with model
            # print(f"[KEEP] High confidence model prediction: {best_intent} ({best_confidence:.2f})")
            return best_intent, best_confidence

        # PRIORITY 2: Handle specific keyword priorities (more specific wins)
        specific_keyword_priorities = {
            # Heartbreak specifics beat general relationship
            'selingkuh': 'heartbreak_cheated',
            'putus': 'heartbreak_breakup',
            'ditolak': 'heartbreak_rejected',
            'ghosting': 'heartbreak_ghosted',
            'menghilang': 'heartbreak_ghosted',
            'entah ke mana': 'heartbreak_ghosted',

            # Academic stress beats general stress
            'tugas': 'stress_due_to_academic',
            'kuliah': 'stress_due_to_academic',
            'skripsi': 'stress_due_to_academic',
            'nilai': 'stress_due_to_academic',
            'ujian': 'stress_due_to_academic',

            # Future anxiety beats general life pressure
            'masa depan': 'anxiety_due_to_future',
            'karir': 'anxiety_due_to_future',

            # Self worth specifics
            'tidak berharga': 'self_worth_general',
            'tidak berguna': 'self_worth_general',
            'minder': 'self_worth_low_confidence',
            'bandingkan': 'self_worth_social_comparison',

            # Sleep issues
            'tidak bisa tidur': 'insomnia_general',
            'susah tidur': 'insomnia_general',

            # Grief specifics - HIGH PRIORITY
            'meninggal': 'grief_loss_of_person',
            'mati': 'grief_loss_of_person',
            'wafat': 'grief_loss_of_person',
            'kecelakaan': 'grief_loss_of_person',
            'sakit parah': 'grief_loss_of_person',
            'rumah sakit': 'grief_loss_of_person',

            # Pet loss specifics
            'kehilangan anjing': 'grief_loss_of_pet',
            'kehilangan kucing': 'grief_loss_of_pet',
            'anjing mati': 'grief_loss_of_pet',
            'kucing mati': 'grief_loss_of_pet',
            'hewan peliharaan': 'grief_loss_of_pet',

            # Loneliness specifics
            'gada teman': 'loneliness_no_friends',
            'ga ada teman': 'loneliness_no_friends',
            'pengen ditemenin': 'loneliness_no_friends',
            'butuh teman': 'loneliness_no_friends'
        }

        # Check for specific high-priority keywords
        for keyword, priority_intent in specific_keyword_priorities.items():
            if keyword in user_lower and priority_intent in intent_scores:
                # Check if this intent is reasonable (in top 3 or has good keyword score)
                if (priority_intent in top_3_intents or
                    intent_scores[priority_intent]['score'] >= 1):  # Lowered from 2 to 1

                    if priority_intent in top_3_intents:
                        idx = top_3_intents.index(priority_intent)
                        original_conf = top_3_confidences[idx]
                        boosted_confidence = min(max(original_conf + 0.4, 0.7), 0.95)  # Stronger boost
                    else:
                        boosted_confidence = min(0.75 + intent_scores[priority_intent]['confidence_boost'], 0.95)

                    # print(f"[PRIORITY] Specific keyword override: {keyword} → {priority_intent} ({boosted_confidence:.2f})")
                    # print(f"[PRIORITY] Keywords: {intent_scores[priority_intent]['keywords']}")
                    return priority_intent, boosted_confidence

        # PRIORITY 3: Enhanced keyword match for bad/medium model predictions
        if best_confidence < 0.6 and top_keyword_data['score'] >= 1:  # Any keyword match for low confidence
            # Filter out problematic intents dari consideration
            problematic_low_conf_intents = ['greet_user', 'bye_user', 'get_name']
            if top_keyword_intent in problematic_low_conf_intents:
                # Look for second best keyword intent
                if len(sorted_intents) > 1:
                    second_intent = sorted_intents[1][0]
                    second_data = sorted_intents[1][1]
                    if second_intent not in problematic_low_conf_intents:
                        boosted_confidence = 0.65 + second_data['confidence_boost']
                        # print(f"[BOOST] Skipped problematic intent, using: {second_intent} ({boosted_confidence:.2f})")
                        # print(f"[BOOST] Keywords: {second_data['keywords']}")
                        return second_intent, min(boosted_confidence, 0.95)

            # Special handling for expectation anxiety - often misclassified
            expectation_keywords = ['dituntut', 'ekspektasi', 'ekspetasi', 'tuntutan', 'pressure']
            if any(keyword in user_lower for keyword in expectation_keywords):
                if 'anxiety_due_to_expectation' in intent_scores:
                    boosted_confidence = 0.75 + intent_scores['anxiety_due_to_expectation']['confidence_boost']
                    # print(f"[EXPECTATION] Special boost for expectation anxiety: anxiety_due_to_expectation ({boosted_confidence:.2f})")
                    return "anxiety_due_to_expectation", min(boosted_confidence, 0.95)

            # Special handling for positive emotions - very important to get right
            positive_keywords = ['senang', 'bahagia', 'gembira', 'excited', 'happy', 'ceria']
            if any(pos_word in user_lower for pos_word in positive_keywords):
                if 'positive_response' in intent_scores:
                    boosted_confidence = 0.8 + intent_scores['positive_response']['confidence_boost']
                    # print(f"[POSITIVE] Special boost for positive emotion: positive_response ({boosted_confidence:.2f})")
                    # print(f"[POSITIVE] Keywords: {intent_scores['positive_response']['keywords']}")
                    return "positive_response", min(boosted_confidence, 0.95)

            # Special handling for grief cases - very sensitive topic
            grief_keywords = ['meninggal', 'mati', 'wafat', 'kecelakaan', 'rumah sakit']
            if any(keyword in user_lower for keyword in grief_keywords):
                if 'grief' in top_keyword_intent:
                    boosted_confidence = 0.8 + top_keyword_data['confidence_boost']
                    # print(f"[GRIEF] Special boost for sensitive topic: {top_keyword_intent} ({boosted_confidence:.2f})")
                    # print(f"[GRIEF] Keywords: {top_keyword_data['keywords']}")
                    return top_keyword_intent, min(boosted_confidence, 0.95)

            # Special handling for loneliness cases - common and important
            loneliness_keywords = ['teman', 'kesepian', 'sendirian', 'lonely', 'ditemenin']
            if any(keyword in user_lower for keyword in loneliness_keywords):
                if 'loneliness' in top_keyword_intent:
                    boosted_confidence = 0.75 + top_keyword_data['confidence_boost']
                    # print(f"[LONELINESS] Special boost for social support: {top_keyword_intent} ({boosted_confidence:.2f})")
                    # print(f"[LONELINESS] Keywords: {top_keyword_data['keywords']}")
                    return top_keyword_intent, min(boosted_confidence, 0.95)

            # Normal low/medium confidence handling - more aggressive
            boosted_confidence = 0.65 + top_keyword_data['confidence_boost']
            # print(f"[BOOST] Low/medium confidence, keyword evidence: {top_keyword_intent} ({boosted_confidence:.2f})")
            # print(f"[BOOST] Keywords: {top_keyword_data['keywords']}")
            return top_keyword_intent, min(boosted_confidence, 0.95)

        # PRIORITY 4: Medium confidence model with keyword support
        elif best_confidence < 0.6 and top_keyword_data['score'] >= 2:
            if top_keyword_intent in top_3_intents:
                idx = top_3_intents.index(top_keyword_intent)
                original_conf = top_3_confidences[idx]
                boosted_confidence = min(original_conf + top_keyword_data['confidence_boost'], 0.95)

                # print(f"[BOOST] Medium confidence with keywords: {top_keyword_intent} ({boosted_confidence:.2f})")
                # print(f"[BOOST] Keywords: {top_keyword_data['keywords']}")
                return top_keyword_intent, boosted_confidence

        # PRIORITY 5: Problematic intent overrides
        problematic_overrides = {
            "grief_due_to_divorce": ["positive_response", "heartbreak_general", "sadness_general"],
            "positive_response": ["self_worth_general", "depression_general", "anxiety_general",
                                 "stress_general", "heartbreak_general", "loneliness_general"]
        }

        if best_intent in problematic_overrides:
            for override_intent in problematic_overrides[best_intent]:
                if override_intent in intent_scores and intent_scores[override_intent]['score'] >= 1:
                    boosted_confidence = 0.6 + intent_scores[override_intent]['confidence_boost']

                    print(f"[OVERRIDE] Problematic prediction: {best_intent} → {override_intent} ({boosted_confidence:.2f})")
                    print(f"[OVERRIDE] Keywords: {intent_scores[override_intent]['keywords']}")

                    return override_intent, min(boosted_confidence, 0.95)

    # Jika tidak ada keyword boost, return original
    return best_intent, best_confidence

def enhanced_predict_intent(user_input):
    """Enhanced prediction dengan keyword boosting dan out-of-domain detection"""

    # Out-of-domain detection dengan exact word matching
    out_of_domain_keywords = [
        'slot', 'gacor', 'betting', 'judi', 'casino', 'poker', 'jackpot',
        'forex', 'trading', 'crypto', 'bitcoin', 'invest', 'saham',
        'jual', 'beli', 'promo', 'diskon', 'murah',
        'toko', 'shop', 'delivery', 'gojek', 'grab', 'ojol'
    ]

    # More specific out-of-domain phrases (exact match)
    out_of_domain_phrases = [
        'online shop', 'king slot', 'slot gacor', 'main slot',
        'beli sekarang', 'promo hari ini', 'diskon besar'
    ]

    user_lower = user_input.lower()

    # Check for exact phrase matches first
    for phrase in out_of_domain_phrases:
        if phrase in user_lower:
            print(f"[OUT-OF-DOMAIN] Detected phrase: {phrase}")
            return "out_of_domain", 0.95, ["out_of_domain"], [0.95]

    # Check for word boundaries to avoid false positives
    import re
    for keyword in out_of_domain_keywords:
        # Use word boundaries to match exact words only
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, user_lower):
            print(f"[OUT-OF-DOMAIN] Detected keyword: {keyword}")
            return "out_of_domain", 0.95, ["out_of_domain"], [0.95]
    # Inappropriate content detection with context awareness
    def is_inappropriate_usage(text_lower):
        """Check if profanity is used inappropriately vs legitimate context"""

        # Context-sensitive words that can be legitimate
        context_sensitive = {
            'anjing': ['hewan', 'peliharaan', 'kehilangan', 'mati', 'meninggal', 'sakit', 'dokter hewan', 'pet'],
            'babi': ['daging', 'makanan', 'masak', 'restoran', 'menu']
        }

        # Always inappropriate (no legitimate context in mental health)
        always_inappropriate = [
            'kepala bapak', 'bangsat', 'tai', 'shit', 'fuck', 'bitch', 'asshole',
            'kontol', 'memek', 'ngentot', 'jancok', 'bajingan', 'tolol', 'goblok'
        ]

        # Check always inappropriate first
        for phrase in always_inappropriate:
            if phrase in text_lower:
                return True, phrase

        # Check context-sensitive words
        for word, valid_contexts in context_sensitive.items():
            if word in text_lower:
                # Check if any valid context is present
                has_valid_context = any(context in text_lower for context in valid_contexts)
                if not has_valid_context:
                    # Also check for obvious insult patterns
                    insult_patterns = [
                        f'kamu {word}', f'lu {word}', f'elu {word}',
                        f'{word} kamu', f'{word} lu', f'{word} banget',
                        f'dasar {word}', f'memang {word}'
                    ]
                    if any(pattern in text_lower for pattern in insult_patterns):
                        return True, word

        return False, None

    # Check for inappropriate content
    is_inappropriate, detected_word = is_inappropriate_usage(user_lower)
    if is_inappropriate:
        print(f"[INAPPROPRIATE] Detected inappropriate usage: {detected_word}")
        return "inappropriate", 0.95, ["inappropriate"], [0.95]

    # Also handle slang/internet terms that might be out of scope
    slang_mental_health = [
        'sigma', 'alpha', 'beta', 'chad', 'virgin', 'based', 'cringe',
        'sus', 'sussy', 'among us', 'imposter', 'cap', 'no cap',
        'skibidi', 'skibdi', 'ohio', 'rizz', 'gyat', 'gyatt', 'fanum',
        'bussin', 'sheesh', 'periodt', 'slay', 'bet', 'fr fr', 'ong',
        'simp', 'npc', 'mid', 'lowkey', 'highkey', 'vibe check'
    ]

    # Check if input is just slang without mental health context
    words = user_lower.split()
    if len(words) <= 2 and any(word in slang_mental_health for word in words):
        # Check if there are any mental health indicators
        mental_health_indicators = [
            'sedih', 'senang', 'cemas', 'takut', 'stress', 'tertekan',
            'depresi', 'galau', 'bingung', 'marah', 'kecewa', 'lelah'
        ]

        if not any(indicator in user_lower for indicator in mental_health_indicators):
            # print(f"[OUT-OF-DOMAIN] Internet slang without mental health context: {user_input}")
            return "out_of_domain", 0.95, ["out_of_domain"], [0.95]

    # Get original model prediction
    processed_input = preprocess_input(user_input)
    seq = tokenizer.texts_to_sequences([processed_input])
    padded = pad_sequences(seq, truncating="post", maxlen=max_len)
    pred = model.predict(padded, verbose=0)[0]

    # Get top 3 predictions
    top_3_indices = np.argsort(pred)[-3:][::-1]
    top_3_intents = [lbl_encoder.inverse_transform([idx])[0] for idx in top_3_indices]
    top_3_confidences = [float(pred[idx]) for idx in top_3_indices]

    original_intent = top_3_intents[0]
    original_confidence = top_3_confidences[0]

    # Apply keyword boosting
    boosted_intent, boosted_confidence = get_keyword_boost(
        user_input, original_intent, top_3_intents, top_3_confidences
    )

    # print(f"[PREDICTION] Original: {original_intent} ({original_confidence:.2f})")
    if boosted_intent != original_intent or boosted_confidence != original_confidence:
        # print(f"[PREDICTION] Boosted: {boosted_intent} ({boosted_confidence:.2f})")

     return boosted_intent, boosted_confidence, top_3_intents, top_3_confidences

# Example usage dalam chatbot_response function:
def enhanced_chatbot_response(user_input):
    """Chatbot response dengan keyword boosting"""

    # Enhanced intent prediction dengan keyword boosting
    tag, confidence, top_3_tags, top_3_confidences = enhanced_predict_intent(user_input)

'''
# Test function
def test_keyword_boosting():
    """Test keyword boosting dengan berbagai input"""
    test_inputs = [
        "aku ngerasa hampa",
        "aku capek",
        "aku ribut sama orang tua",
        "aku ribut sama pacar",
        "aku gak sanggup sama hidup ini",
        "Aku capek banget dituntut orang lain",
        "aku merasa tidak berguna dan ingin mati",
        "aku sedih karena ditolak",
        "dosenku bikin aku tertekan",
        "anjingku mati kemarin",
        "orang tuaku cerai"
    ]

    for inp in test_inputs:
        print(f"\n{'='*50}")
        print(f"Input: {inp}")
        enhanced_predict_intent(inp)

if __name__ == "__main__":
    test_keyword_boosting()
'''


# === Load Model Components ===
model = load_model("chatbot_model_v1.h5")
# print("[SUCCESS] Loaded chatbot_model_v1.h5")

with open("tokenizer_v1.pickle", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder_v1.pickle", "rb") as f:
    lbl_encoder = pickle.load(f)

max_len = 25

# === Global State Management ===
class ChatbotState:
    def __init__(self):
        self.context = None
        self.user_name = ""
        self.fail_count = 0
        self.current_topic = None
        self.last_tag = None
        self.conversation_history = []
        self.session_start = True

    def reset(self):
        old_name = self.user_name
        self.__init__()
        return f"🔄 Obrolan telah direset. Sampai jumpa {old_name}! 😊"

    def add_to_history(self, user_input, bot_response, intent=None, confidence=None):
        self.conversation_history.append({
            'user': user_input,
            'bot': bot_response,
            'intent': intent,
            'confidence': confidence,
            'context': self.context
        })

# Init state
state = ChatbotState()

# === Constants ===
MAX_FAILS = 3
CONFIDENCE_THRESHOLD = 0.5

# === Intent Categories ===
allowed_general_intents = {
    "stress_general", "anxiety_general", "self_worth_general",
    "heartbreak_general", "loneliness_general", "grief_general",
    "depression_general", "badmood_general", "galau_general", "insomnia_general",
    "overthinking_general", "sadness_general"
}

universal_intents = {
    "get_support_professional", "greet_user", "bye_user", "positive_response"
}

# === Import your keyword mapping and enhanced prediction ===

# === Load Responses from intents_main.json ===
import json

# Load responses dari intents_main.json
try:
    with open("intents_main.json", "r", encoding="utf-8") as f:
        intents_data = json.load(f)

    # Extract responses untuk setiap intent
    responses = {}
    for intent in intents_data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]

    # print("[SUCCESS] Loaded responses from intents_main.json")

except FileNotFoundError:
    # print("[ERROR] intents_main.json not found, using default responses")
    # Fallback responses jika file tidak ditemukan
    responses = {
        "greet_user": [
            "Halo! Senang bertemu denganmu! Aku di sini untuk mendengarkan dan mendukungmu. Siapa namamu?",
            "Hai! Gimana kabarmu hari ini? Boleh aku tahu namamu?",
            "Selamat datang! Aku di sini untuk menemanimu. Siapa yang sedang aku ajak bicara?"
        ],
        "positive_response": [
            "Senang sekali mendengar itu! 😊 Syukurlah kamu merasa baik hari ini.",
            "Wah, bagus banget! Senang lihat kamu dalam mood yang baik.",
            "Alhamdulillah ya! Semoga perasaan baikmu ini terus berlanjut."
        ],
    }

# Import tips_responses dan decline_responses

def get_response_from_intent(tag):
    """Get random response from intents_main.json responses"""
    if tag in responses and len(responses[tag]) > 0:
        return random.choice(responses[tag]).replace("{user_name}", state.user_name or "kamu")
    return f"Aku mendengar apa yang kamu katakan, {state.user_name or 'kamu'}."

def get_general_response_with_options(tag):
    """Get general response with multiple choice options"""

    # Dapatkan base response dari intents_main.json
    base_response = get_response_from_intent(tag)

    # Tambahan pilihan spesifik untuk general intents
    options_text = {
        "stress_general": "\n\nUntuk bisa membantumu dengan lebih tepat, bisa kamu ceritakan sedikit lebih detail tentang sumber stressmu?\n\n1. 📚 Tekanan akademik (kuliah, tugas, ujian)\n2. 💼 Masalah pekerjaan atau karir\n3. 👨‍👩‍👧‍👦 Konflik atau tekanan dari keluarga\n4. 💔 Masalah dalam hubungan percintaan\n5. 🌍 Tekanan hidup secara umum (finansial, masa depan, dll)",

        "anxiety_general": "\n\nAgar aku bisa memahami lebih baik, bisa kamu pilih mana yang paling mendekati kecemasanmu?\n\n1. 🔮 Cemas tentang masa depan atau hal yang belum terjadi\n2. 😰 Takut gagal atau tidak memenuhi ekspektasi\n3. 📈 Merasa tertekan karena ekspektasi tinggi\n4. 👥 Cemas dalam situasi sosial atau bertemu orang",

        "depression_general": "\n\nBisa kamu ceritakan lebih lanjut tentang apa yang kamu rasakan?\n\n1. 😔 Kesedihan yang berkepanjangan atau terus-menerus\n2. 😶 Hilang minat pada hal-hal yang dulu menyenangkan\n3. 😴 Merasa mati rasa atau kosong secara emosional",

        "heartbreak_general": "\n\nBisa kamu ceritakan lebih detail tentang situasimu?\n\n1. 💔 Baru saja putus atau berpisah dari pasangan\n2. 🔥 Merasa dikhianati karena diselingkuhi\n3. 💧 Perasaan ditolak atau tidak diterima\n4. 👻 Ditinggalkan tanpa penjelasan (di-ghosting)",

        "loneliness_general": "\n\nMana yang paling menggambarkan perasaanmu?\n\n1. 👥 Merasa tidak punya teman atau sulit bersosialisasi\n2. 🗣️ Tidak ada orang yang bisa diajak bicara atau curhat\n3. 🌫️ Merasa tidak dipahami oleh orang-orang di sekitar",

        "grief_general": "\n\nBisa kamu ceritakan tentang kehilangan yang sedang kamu rasakan?\n\n1. 😢 Kehilangan orang yang dicintai (meninggal dunia)\n2. 🐕 Kehilangan hewan peliharaan tersayang\n3. 💍 Berduka karena perceraian orang tua",

        "self_worth_general": "\n\nMana yang paling menggambarkan perasaanmu tentang diri sendiri?\n\n1. 😟 Kurang percaya diri dan sering merasa minder\n2. 📊 Sering membandingkan diri dengan orang lain\n3. 🎭 Merasa seperti penipu atau tidak pantas (imposter syndrome)",

        "overthinking_general": "\n\nApa yang paling sering kamu pikirkan berlebihan?\n\n1. 🤔 Keputusan-keputusan yang harus diambil\n2. 💕 Hal-hal yang berkaitan dengan hubungan\n3. 🪞 Diri sendiri dan kehidupanmu secara umum"
    }

    if tag in options_text:
        return base_response + options_text[tag]
    else:
        return base_response
def extract_name(text):
    """Enhanced name extraction with better patterns"""
    text = text.lower().strip()

    patterns = [
        r"namaku\s+([a-zA-Z]+)",
        r"nama\s+saya\s+([a-zA-Z]+)",
        r"nama\s+aku\s+([a-zA-Z]+)",
        r"saya\s+([a-zA-Z]+)",
        r"aku\s+([a-zA-Z]+)",
        r"panggil\s+aku\s+([a-zA-Z]+)",
        r"([a-zA-Z]+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1)
            common_words = {'saya', 'aku', 'kamu', 'dia', 'mereka', 'kami', 'kita'}
            if name.lower() not in common_words and len(name) > 1:
                return name.capitalize()

    return "Teman"

def handle_appreciation(user_input):
    """Handle various forms of appreciation"""
    appreciation_patterns = [
        'makasih', 'terima kasih', 'thanks', 'thank you', 'tengkyu',
        'tq', 'thx', 'makasi', 'trimakasi'
    ]

    return any(pattern in user_input.lower() for pattern in appreciation_patterns)

def handle_greetings(user_input):
    """Enhanced greeting detection"""
    greeting_patterns = [
        'hai', 'halo', 'hello', 'hi', 'hey', 'selamat pagi',
        'selamat siang', 'selamat sore', 'selamat malam'
    ]

    return any(pattern in user_input.lower() for pattern in greeting_patterns)

def get_general_options_mapping():
    """Mapping pilihan user ke specific intent"""
    return {
        "stress_general": {
            "1": "stress_due_to_academic",
            "2": "stress_due_to_work",
            "3": "stress_due_to_family",
            "4": "stress_due_to_relationship",
            "5": "stress_due_to_life_pressure"
        },
        "anxiety_general": {
            "1": "anxiety_due_to_future",
            "2": "anxiety_due_to_failure",
            "3": "anxiety_due_to_expectation",
            "4": "anxiety_due_to_social"
        },
        "depression_general": {
            "1": "depression_chronic_sadness",
            "2": "depression_loss_of_interest",
            "3": "depression_emotional_numbness"
        },
        "heartbreak_general": {
            "1": "heartbreak_breakup",
            "2": "heartbreak_cheated",
            "3": "heartbreak_rejected",
            "4": "heartbreak_ghosted"
        },
        "loneliness_general": {
            "1": "loneliness_no_friends",
            "2": "loneliness_no_one_to_talk_to",
            "3": "loneliness_feel_misunderstood"
        },
        "grief_general": {
            "1": "grief_loss_of_person",
            "2": "grief_loss_of_pet",
            "3": "grief_due_to_divorce"
        },
        "self_worth_general": {
            "1": "self_worth_low_confidence",
            "2": "self_worth_social_comparison",
            "3": "self_worth_imposter_syndrome"
        },
        "overthinking_general": {
            "1": "overthinking_about_decision",
            "2": "overthinking_about_relationship",
            "3": "overthinking_about_self"
        }
    }

def get_fallback_response():
    """Enhanced fallback responses"""
    fallback_responses = [
        f"Hmm... aku belum yakin maksudmu, {state.user_name or 'teman'} 😅 Bisa dijelaskan dengan cara lain?",
        f"Aku agak bingung nangkap maksud kamu, {state.user_name or 'teman'}. Bisa diperjelas?",
        f"Maaf {state.user_name or 'teman'}, aku belum paham betul. Bisa diulangi dengan kalimat yang berbeda?",
        f"Sepertinya aku perlu penjelasan lebih, {state.user_name or 'teman'}. Bisa cerita lebih detail?"
    ]
    return random.choice(fallback_responses)

def get_specific_tip(tag):
    """Get specific tip response from tips_responses"""
    if tag in tips_responses:
        response = random.choice(tips_responses[tag])
        return response.replace("{user_name}", state.user_name or "kamu")
    return None

def get_specific_decline(tag):
    """Get specific decline response from decline_responses"""
    if tag in decline_responses:
        response = random.choice(decline_responses[tag])
        return response.replace("{user_name}", state.user_name or "kamu")
    return None

# === Main Chatbot Response Function ===
def is_inline_emotion_tag(tag):
    return tag in [
        "positive_responses",
        "sadness_general",
        "galau_general",
        "insomnia_general"
    ]

def is_inline_emotion_tag(tag):
    return tag in [
        "positive_responses",
        "sadness_general",
        "galau_general",
        "insomnia_general"
    ]

# Simulasi variabel sesi pengguna
current_topic = None
last_tag = None
context = None
emotion_context = None

def chatbot_response(user_input):
    original_input = user_input.strip()
    user_input_lower = user_input.lower().strip()

    if user_input_lower in ["reset", "ulang", "mulai lagi", "start over"]:
        return state.reset()

    if handle_appreciation(user_input_lower):
        appreciation_responses = [
            f"Sama-sama ya, {state.user_name or 'teman'} 🌷 Senang bisa bantu.",
            f"Dengan senang hati, {state.user_name or 'teman'} 💙 Aku selalu di sini untukmu.",
            f"Tidak apa-apa, {state.user_name or 'teman'} ✨ Terima kasih sudah mau berbagi denganku.",
            f"Senang bisa menemani, {state.user_name or 'teman'} 🤗 Jangan ragu cerita lagi kalau butuh ya."
        ]
        response = random.choice(appreciation_responses)
        state.add_to_history(original_input, response)
        return response

    if state.context is None:
        state.context = "awaiting_name"
        if handle_greetings(user_input_lower):
            response = f"Hai! Senang bertemu denganmu! ✨ Aku di sini untuk mendengarkan dan mendukungmu. Boleh aku tahu siapa namamu?"
        else:
            response = "Hai! Boleh aku tahu siapa namamu? 😊"
        state.add_to_history(original_input, response)
        return response

    if state.context == "awaiting_name":
        state.user_name = extract_name(user_input)
        state.context = "awaiting_feeling"
        response = f"Hai {state.user_name}! Senang berkenalan denganmu 😊\n\nGimana perasaanmu hari ini? Apa yang sedang ada di pikiran atau hatimu yang ingin kamu ceritakan?"
        state.add_to_history(original_input, response)
        return response

    if state.context == "awaiting_tip_permission":
        if user_input_lower in ["iya", "ya", "mau", "boleh", "lanjut", "oke", "ok", "yup", "yes", "y"]:
            state.context = "conversation_end"
            tip_response = get_specific_tip(state.last_tag)
            response = tip_response or f"🙏 Maaf {state.user_name}, belum ada tips khusus untuk itu, tapi aku tetap di sini untuk mendengarkanmu."
        elif user_input_lower in ["tidak", "ga", "gak", "nggak", "skip", "gausah", "enggak", "no", "n"]:
            state.context = "conversation_end"
            decline_response = get_specific_decline(state.last_tag)
            response = decline_response or f"Oke {state.user_name}, aku tetap di sini kalau kamu butuh. Tidak apa-apa kalau kamu belum siap 🤍"
        else:
            response = f"Maaf {state.user_name}, aku belum paham jawabanmu. Mau aku kasih tips atau tidak? Jawab 'Ya' atau 'Tidak' ya 😊"

        state.add_to_history(original_input, response)
        return response

    if state.context == "conversation_end":
        state.context = "awaiting_feeling"
        state.current_topic = None
        state.last_tag = None

    if state.context == "awaiting_choice":
        choice_mapping = get_general_options_mapping()
        if state.current_topic in choice_mapping:
            valid_choices = choice_mapping[state.current_topic]
            if user_input_lower in valid_choices:
                specific_intent = valid_choices[user_input_lower]
                state.context = "awaiting_tip_permission"
                state.last_tag = specific_intent

                base_response = get_response_from_intent(specific_intent)
                full_response = f"{base_response}\n\nMau aku bantu kasih beberapa tips untuk menghadapi ini, {state.user_name}?"

                state.add_to_history(original_input, full_response, specific_intent, 1.0)
                return {
                    "text": full_response,
                    "options": ["Ya", "Tidak"]
                }
            else:
                response = f"Pilih nomor 1-{len(valid_choices)} ya, {state.user_name} 😊"
                state.add_to_history(original_input, response)
                return response

    try:
        prediction_result = enhanced_predict_intent(user_input)

        if prediction_result is None or not isinstance(prediction_result, (tuple, list)) or len(prediction_result) != 4:
            processed_input = preprocess_input(user_input)
            seq = tokenizer.texts_to_sequences([processed_input])
            padded = pad_sequences(seq, truncating="post", maxlen=max_len)
            pred = model.predict(padded, verbose=0)[0]
            top_idx = np.argmax(pred)
            tag = lbl_encoder.inverse_transform([top_idx])[0]
            confidence = float(pred[top_idx])
            top_3_tags = [tag]
            top_3_confidences = [confidence]
        else:
            tag, confidence, top_3_tags, top_3_confidences = prediction_result

    except Exception as e:
        response = f"Maaf {state.user_name or 'teman'}, ada sedikit gangguan teknis. Bisa coba lagi?"
        state.add_to_history(original_input, response)
        return response

    if is_inline_emotion_tag(tag):
        state.emotion_context = tag
        response = get_response_from_intent(tag)
        state.add_to_history(original_input, response, tag, confidence)
        return response

    invalid_transitions = [
    (a, b)
    for a in ["positive_responses", "sadness_general", "galau_general", "insomnia_general"]
    for b in ["positive_responses", "sadness_general", "galau_general", "insomnia_general"]
    if a != b
]
    if hasattr(state, 'emotion_context') and state.emotion_context and (state.emotion_context, tag) in invalid_transitions:
        response = (
            f"Tadi kamu sempat merasa {state.emotion_context.replace('_general','').replace('_',' ')}, "
            f"sekarang kamu bilang {tag.replace('_general','').replace('_',' ')}. "
            "Maukah kamu cerita lebih lanjut supaya aku bisa memahami lebih dalam?"
        )
        state.add_to_history(original_input, response, tag, confidence)
        return response
        response = (
            f"Tadi kamu sempat merasa {emotion_context.replace('_general','').replace('_',' ')}, "
            f"sekarang kamu bilang {tag.replace('_general','').replace('_',' ')}. "
            "Maukah kamu cerita lebih lanjut supaya aku bisa memahami lebih dalam?"
        )
        state.add_to_history(original_input, response, tag, confidence)
        return response

    if tag in allowed_general_intents and state.context != "awaiting_choice":
        state.context = "awaiting_choice"
        state.current_topic = tag
        state.emotion_context = None  # Reset emosi saat mulai topik baru
        response = get_general_response_with_options(tag)
        state.add_to_history(original_input, response, tag, confidence)
        return response

    if tag == "out_of_domain":
        response = f"Maaf {state.user_name or 'teman'}, sepertinya topik ini di luar area yang bisa aku bantu. Aku di sini khusus untuk mendengarkan dan mendukung kesehatan mentalmu. Ada hal lain yang ingin kamu ceritakan tentang perasaan atau situasi yang sedang kamu hadapi?"
        state.add_to_history(original_input, response, tag, confidence)
        return response

    if tag == "inappropriate":
        response = f"Maaf {state.user_name or 'teman'}, aku tidak bisa merespons kata-kata yang kasar atau tidak sopan. Aku di sini untuk mendukungmu dengan cara yang positif. Apakah ada hal lain yang ingin kamu ceritakan tentang perasaanmu?"
        state.add_to_history(original_input, response, tag, confidence)
        return response

    if tag in universal_intents:
        if tag == "greet_user":
            response = f"Hai {state.user_name}! Gimana kabarmu hari ini?"
        elif tag == "bye_user":
            response = responses["bye_user"][0].replace("{user_name}", state.user_name or "teman")
        elif tag == "positive_response":
            response = f"Senang dengar itu, {state.user_name}! 😊 Syukurlah kamu merasa baik. Ada yang ingin kamu ceritakan lebih lanjut?"
        else:
            response = responses.get(tag, [f"Kamu bisa cari bantuan profesional ya, {state.user_name}"])[0]

        state.add_to_history(original_input, response, tag, confidence)
        return response

    if state.context == "awaiting_feeling" and tag in allowed_general_intents:
        state.current_topic = tag
        state.context = "awaiting_choice"
        response = get_general_response_with_options(tag)
        state.add_to_history(original_input, response, tag, confidence)
        return response

    if confidence < CONFIDENCE_THRESHOLD:
        state.fail_count += 1
        if state.fail_count >= MAX_FAILS:
            response = state.reset()
            state.add_to_history(original_input, response)
            return response

        response = get_fallback_response()
        state.add_to_history(original_input, response, tag, confidence)
        return response

    if tag.endswith((
        '_due_to_academic', '_due_to_work', '_due_to_family', '_due_to_relationship', '_due_to_life_pressure',
        '_due_to_future', '_due_to_failure', '_due_to_expectation', '_due_to_social',
        '_low_confidence', '_social_comparison', '_imposter_syndrome',
        '_breakup', '_cheated', '_rejected', '_ghosted',
        '_no_friends', '_no_one_to_talk_to', '_feel_misunderstood',
        '_loss_of_person', '_loss_of_pet', '_due_to_divorce',
        '_chronic_sadness', '_loss_of_interest', '_emotional_numbness',
        '_about_decision', '_about_relationship', '_about_self')):

        state.context = "awaiting_tip_permission"
        state.last_tag = tag
        base_response = get_response_from_intent(tag)
        full_response = f"{base_response}\n\nMau aku bantu kasih beberapa tips untuk menghadapi ini, {state.user_name}?"
        state.add_to_history(original_input, full_response, tag, confidence)
        return {
            "text": full_response,
            "options": ["Ya", "Tidak"]
        }

    response = get_response_from_intent(tag)
    state.fail_count = 0
    state.add_to_history(original_input, response, tag, confidence)
    return response




# === Interaksi chatbot ===
def main():
    # print("🤖 Mental Health Support Chatbot")
    # print("💚 Aku di sini untuk mendengarkan dan mendukungmu")
    # print("💬 Ketik 'quit' untuk keluar atau 'reset' untuk memulai ulang")
    # print("=" * 50)

    while True:
        try:
            inp = input("\n💭 You: ")

            if inp.lower() in ["quit", "exit", "keluar"]:
                if state.user_name:
                    print(f"🤖 Bot: Terima kasih sudah berbagi hari ini, {state.user_name}. Jaga diri baik-baik ya! Aku akan selalu di sini kalau kamu butuh. Sampai jumpa! 💚")
                else:
                    print("🤖 Bot: Sampai jumpa! Jaga diri baik-baik ya! 💚")
                break

            if inp.lower() in ["stats", "statistik"]:
                stats = {
                    "total_exchanges": len(state.conversation_history),
                    "current_context": state.context,
                    "user_name": state.user_name,
                    "current_topic": state.current_topic,
                    "fail_count": state.fail_count
                }
                print(f"📊 Stats: {stats}")
                continue

            response = chatbot_response(inp)

            if isinstance(response, dict):
                print(f"🤖 Bot: {response['text']}")
                if 'options' in response:
                    print(f"💡 Pilihan: {' | '.join(response['options'])}")
            else:
                print(f"🤖 Bot: {response}")

        except KeyboardInterrupt:
            if state.user_name:
                print(f"\n🤖 Bot: Sampai jumpa {state.user_name}! 💚")
            else:
                print("\n🤖 Bot: Sampai jumpa! 💚")
            break
        except Exception as e:
            print(f"🤖 Bot: Maaf, ada kesalahan teknis. {str(e)}")
            print("🤖 Bot:", state.reset())

# === Endpoint Flask ===
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    if not user_message:
        return jsonify({"response": "Pesan tidak boleh kosong"}), 400
    bot_reply = chatbot_response(user_message)
    return jsonify({"response": bot_reply})

# === Jalankan Server ===
if __name__ == "__main__":
    # Autentikasi ngrok


    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
