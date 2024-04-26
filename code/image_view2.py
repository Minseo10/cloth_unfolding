import threading
import tkinter as tk
from PIL import Image, ImageTk
import os
import open3d as o3d

class ImageViewer:
    def __init__(self, root, start_index=0, end_index=1000):
        self.root = root
        self.start_index = start_index
        self.end_index = end_index
        self.current_index = start_index

        # 이미지 레이블을 담을 리스트 생성
        self.image_labels = []

        # 이미지 레이블을 세 개 생성하고 루트 윈도우에 추가
        for i in range(3):
            image_label = tk.Label(root)
            image_label.grid(row=i, column=0, padx=5, pady=5)  # 그리드 레이아웃에서 행 0, 열 i에 배치
            self.image_labels.append(image_label)

        # 경로 레이블 생성
        self.path_label = tk.Label(root, text="")
        self.path_label.grid(row=0, column=1, columnspan=3)

        # 이전
        self.next_button = tk.Button(root, text="<<<", command=self.prev_directory)
        self.next_button.grid(row=1, column=1, pady=10)

        # "다음" 버튼 생성
        self.next_button = tk.Button(root, text=">>>", command=self.next_directory)
        self.next_button.grid(row=1, column=2, pady=10)

        # 첫 번째 디렉토리의 이미지를 표시
        self.show_images()

    def show_images(self):
        # 이미지 파일 경로
        base_path = f"../datasets/cloth_competition_dataset_0000/sample_{'{0:06d}'.format(self.current_index)}"

        # 각 이미지 파일을 읽어와 레이블에 표시
        for i in range(3):
            image_path = os.path.join(base_path, "processing", f"example_grasp_pose_{i}.png")
            if os.path.exists(image_path):
                # 이미지 읽기
                img = Image.open(image_path)
                # Tkinter에서 사용할 수 있는 형식으로 변환
                img_tk = ImageTk.PhotoImage(img)
                # 해당 레이블에 이미지 설정
                self.image_labels[i].configure(image=img_tk)
                # 이미지 참조를 유지
                self.image_labels[i].image = img_tk

        # 현재 디렉토리 경로를 path_label에 표시
        self.path_label.configure(text=f"현재 경로: {base_path}")
        self.show_pcd()

    def next_directory(self):
        # 현재 인덱스 증가
        self.current_index += 1

        # 새로운 인덱스가 end_index를 초과하는 경우
        if self.current_index > self.end_index:
            self.root.quit()
            return

        # 다음 디렉토리의 이미지를 표시
        self.show_images()

    def prev_directory(self):
        # 현재 인덱스 증가
        self.current_index -= 1

        # 새로운 인덱스가 end_index를 초과하는 경우
        if self.current_index < 0:
            self.root.quit()
            return

        # 다음 디렉토리의 이미지를 표시
        self.show_images()
        self.show_pcd()

    def show_pcd(self):
        pcd_path = f"../datasets/cloth_competition_dataset_0000/sample_{'{0:06d}'.format(self.current_index)}/processing/edges.ply"
        edge_pcd = o3d.io.read_point_cloud(pcd_path)
        o3d.visualization.draw_geometries([edge_pcd])


# Tkinter 루트 창 생성
root = tk.Tk()
root.title("이미지 뷰어")

# ImageViewer 클래스 인스턴스 생성
viewer = ImageViewer(root, start_index=1, end_index=1000)

# Tkinter 메인 루프 시작
root.mainloop()
