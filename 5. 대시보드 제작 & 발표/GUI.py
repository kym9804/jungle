import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt, QUrl


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sequence = []
        self.cluster_map = {
            'A': ['C', 'D'],
            'B': ['C', 'D'],
            'C': ['E', 'F'],
            'D': ['E', 'F'],
            'E': ['G', 'H'],
            'F': ['G', 'H'],
        }
        self.clusters = {
            '중꺾마': [['A', 'C', 'E', 'G'], ['A', 'D', 'E', 'G'], ['A', 'C', 'E', 'H'], ['A', 'D', 'E', 'H']],
            '육식': [['B', 'C', 'E', 'G'], ['B', 'C', 'F', 'G'], ['B', 'D', 'E', 'G'], ['B', 'D', 'F', 'G']],
            '베어그릴스': [['B', 'C', 'E', 'H'], ['B', 'C', 'F', 'H'], ['B', 'D', 'E', 'H'], ['B', 'D', 'F', 'H']],
            'AP': [['A', 'C', 'F', 'G'], ['A', 'D', 'F', 'G'], ['A', 'C', 'F', 'H'], ['A', 'D', 'F', 'H']],
        }
        self.initUI()

    def initUI(self):
        self.setWindowTitle('정글러 성격 테스트')
        self.showFullScreen()  # 창을 전체 화면으로 표시
        self.home()

    def home(self):
        self.sequence.clear()

        self.central_widget = QWidget()
        self.layout = QVBoxLayout()

        # Load and display the image
        self.image_label = QLabel(self)
        pixmap = QPixmap('정글차이.png')
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)  # Ensures the image fits within the QLabel
        self.layout.addWidget(self.image_label)

        # # 라벨 중앙 정렬 및 스타일 설정
        # self.label = QLabel("다음 중 선호하는 플레이를 골라주세요", self)
        # self.label.setAlignment(Qt.AlignCenter)  # 중앙 정렬
        # self.label.setStyleSheet("color: black; font-size: 18px;")  # 글씨 색: 빨간색, 크기: 18px
        # self.layout.addWidget(self.label)

        self.start_button = QPushButton('시작', self)
        self.start_button.clicked.connect(self.start)
        self.layout.addWidget(self.start_button)

        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

    def start(self):
        self.central_widget = QWidget()
        self.layout = QVBoxLayout()

        # self.label = QLabel("다음 중 선호하는 플레이를 골라주세요\n아군 정글 우선 vs 적 정글 뚫기", self)
        # self.label.setAlignment(Qt.AlignCenter)  # 라벨을 가운데 정렬
        # self.layout.addWidget(self.label)
        #
        self.video_layout = QHBoxLayout()

        # 왼쪽 YouTube 동영상
        self.left_video_widget = QWebEngineView()
        self.left_video_widget.setUrl(QUrl("https://www.youtube.com/embed/wir-MQyGqLk"))
        self.video_layout.addWidget(self.left_video_widget)

        # 오른쪽 YouTube 동영상
        self.right_video_widget = QWebEngineView()
        self.right_video_widget.setUrl(QUrl("https://www.youtube.com/embed/PpJYs_y6Mlc"))
        self.video_layout.addWidget(self.right_video_widget)

        self.layout.addLayout(self.video_layout)

        self.button_layout = QHBoxLayout()

        self.buttonA = QPushButton('아군 정글 우선', self)
        self.buttonA.clicked.connect(lambda: self.navigate('A'))
        self.button_layout.addWidget(self.buttonA)

        self.buttonB = QPushButton('적 정글 뚫기', self)
        self.buttonB.clicked.connect(lambda: self.navigate('B'))
        self.button_layout.addWidget(self.buttonB)

        self.layout.addLayout(self.button_layout)

        home_button = QPushButton('처음으로', self)
        home_button.clicked.connect(self.home)
        self.layout.addWidget(home_button)

        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

    def navigate(self, letter):
        self.sequence.append(letter)
        if len(self.sequence) == 4:
            self.display_cluster()
        else:
            self.show_next_buttons()

    def show_next_buttons(self):
        current_letter = self.sequence[-1]
        next_buttons = self.cluster_map.get(current_letter, [])

        self.central_widget = QWidget()
        self.layout = QVBoxLayout()

        self.video_layout = QHBoxLayout()

        # A 또는 B를 눌렀을 때 표시할 동영상 URL
        if current_letter in ['A', 'B']:
            left_video_url = "https://www.youtube.com/embed/wu-MfqLBLts?start=15"
            right_video_url = "https://www.youtube.com/embed/dTpcaU_g00s"
            button_texts = ['한타로 승리', '공성으로 승리']
        elif current_letter in ['C', 'D']:
            left_video_url = "https://www.youtube.com/embed/7A0lllum_wQ?start=12"
            right_video_url = 'https://www.youtube.com/embed/A1qkTBSlNmc?start=770'
            button_texts = ['리신의 인섹킥', '카서스의 진혼곡']
        elif current_letter in ['E', 'F']:
            left_video_url = "https://www.youtube.com/embed/39Jbu8FShq0?start=9"
            right_video_url = "https://www.youtube.com/embed/x7mwwJVXnYI"
            button_texts = ['피지컬 믿고 교전', '뇌지컬 믿고 수싸움']

        # 왼쪽 YouTube 동영상
        self.left_video_widget = QWebEngineView()
        self.left_video_widget.setUrl(QUrl(left_video_url))
        self.video_layout.addWidget(self.left_video_widget)

        # 오른쪽 YouTube 동영상
        self.right_video_widget = QWebEngineView()
        self.right_video_widget.setUrl(QUrl(right_video_url))
        self.video_layout.addWidget(self.right_video_widget)

        self.layout.addLayout(self.video_layout)

        self.button_layout = QHBoxLayout()

        button1 = QPushButton(button_texts[0], self)
        button1.clicked.connect(lambda: self.navigate(next_buttons[0]))
        self.button_layout.addWidget(button1)

        button2 = QPushButton(button_texts[1], self)
        button2.clicked.connect(lambda: self.navigate(next_buttons[1]))
        self.button_layout.addWidget(button2)

        self.layout.addLayout(self.button_layout)

        back_button = QPushButton('뒤로가기', self)
        back_button.clicked.connect(self.back)
        self.layout.addWidget(back_button)

        home_button = QPushButton('처음으로', self)
        home_button.clicked.connect(self.home)
        self.layout.addWidget(home_button)

        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

    def back(self):
        if self.sequence:
            self.sequence.pop()
            if not self.sequence:
                self.home()
            else:
                self.show_next_buttons()
        else:
            self.home()

    def display_cluster(self):
        cluster_number = self.find_cluster(self.sequence)

        image_paths = {
            '중꺾마': '중꺾마형.png',
            '육식': '육식형.png',
            '베어그릴스': '베어그릴스형.png',
            'AP': 'AP형.png'
        }

        image_path = image_paths.get(cluster_number, None)

        if image_path:
            self.central_widget = QWidget()
            self.layout = QVBoxLayout()

            # Load and display the image based on the cluster number
            self.image_label = QLabel(self)
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)  # Ensures the image fits within the QLabel
            self.layout.addWidget(self.image_label)

            home_button = QPushButton('처음으로', self)
            home_button.clicked.connect(self.home)
            self.layout.addWidget(home_button)

            self.central_widget.setLayout(self.layout)
            self.setCentralWidget(self.central_widget)

    def find_cluster(self, sequence):
        for cluster_number, paths in self.clusters.items():
            if sequence in paths:
                return cluster_number
        return 'Unknown'


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
