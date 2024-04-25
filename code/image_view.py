import tkinter as tk
from PIL import Image, ImageTk
import os


class ImageViewer:
    def __init__(self, root, start_index=1, end_index=1000):
        self.root = root
        self.start_index = start_index
        self.end_index = end_index
        self.current_index = start_index

        # 이미지 레이블 생성
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # 경로 레이블 생성
        self.path_label = tk.Label(root, text="")
        self.path_label.pack()

        # "닫기" 버튼 생성
        self.close_button = tk.Button(root, text="닫기", command=self.close_image)
        self.close_button.pack()

        # 첫 번째 이미지 표시
        self.show_image()

    def show_image(self):
        if self.current_index > self.end_index:
            self.root.quit()
            return

        # 이미지 파일 경로
        image_path = f"../datasets/cloth_competition_dataset_0001/sample_{'{0:06d}'.format(self.current_index)}/observation_start/image_left.png"

        # 이미지가 존재하는지 확인
        if os.path.exists(image_path):
            # 이미지 읽기
            img = Image.open(image_path)
            # Tkinter에서 사용할 수 있는 형식으로 변환
            img_tk = ImageTk.PhotoImage(img)
            # 레이블에 이미지 설정
            self.image_label.configure(image=img_tk)
            # 이미지 참조를 유지
            self.image_label.image = img_tk

            # 현재 보고 있는 이미지의 파일 경로를 path_label에 표시
            self.path_label.configure(text=f"현재 경로: {image_path}")

        # 현재 인덱스 증가
        self.current_index += 1

    def close_image(self):
        # "닫기" 버튼이 클릭되면 다음 이미지를 표시
        self.show_image()


# Tkinter 루트 창 생성
root = tk.Tk()
root.title("이미지 뷰어")

# ImageViewer 클래스 인스턴스 생성
viewer = ImageViewer(root, start_index=1, end_index=1000)

# Tkinter 메인 루프 시작
root.mainloop()