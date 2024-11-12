import numpy as np
import os
import cv2

class transformation_base(object):
    def load_ckpt(self):
        pass

    def save_ckpt(self):
        pass

    def __init__(self):
        self.init_calibration_frame_count = 0
        self.init_calibrating_pins        = []
        self.init_calibration_done        = True

    def init_calibrate(self):
        pass

    def incr_calibrate(self):
        pass

    def transform(self,src_left,src_right):
        return (src_left + src_right) / 2

class transformation_affine(transformation_base):
    def load_ckpt(self,filename):
        if os.path.exists(filename):
            try:
                data = np.loadtxt(filename)
                if data.shape == (2, 6):
                    matrix1 = data[:, :3]
                    matrix2 = data[:, 3:]
                    return True, np.array(matrix1), np.array(matrix2)
                else:
                    print("Calibration Checkpoint file: Incorrect format!")
            except Exception as e:
                print(f"Calibration Checkpoint reading error: {e}")
        else:
            print("Calibration Checkpoint does not exists, will perform initial calibration first")
        return False, None, None

    def save_ckpt(self,filename, matrix1, matrix2):
        if (isinstance(matrix1, np.ndarray) and matrix1.shape == (2, 3) and
        isinstance(matrix2, np.ndarray) and matrix2.shape == (2, 3)):
            try:
                combined_matrix = np.hstack((matrix1, matrix2))
                np.savetxt(filename, combined_matrix)
                print("Saved to Calibration Checkpoint")
            except Exception as e:
                print(f"Calibration Checkpoint writting error: {e}")
        else:
            print("Matrix data: Incorrect format!")

    def set_Affine(self,Affine_M_left,Affine_M_right):
        self.Affine_M_left  = Affine_M_left 
        self.Affine_M_right = Affine_M_right
    
    def __init__(self):
        self.init_calibration_frame_count = 90
        self.init_calibrating_pins        = [[0.9,0.5],[0.1,0.1],[0.1,0.9]]
        self.init_calibration_done        = False
        self.checkpoint_name              = "Cali_ckpt_affine.txt"

        # Check if checkpoint exists (if exists can skip init calibration)
        self.init_calibration_done = False
        # Transformation
        pts = np.float32([[0,0],[0,1],[1,0]])
        self.set_Affine(cv2.getAffineTransform(pts, pts),cv2.getAffineTransform(pts, pts))

        loaded, affine_matrix_left,affine_matrix_right  = self.load_ckpt(self.checkpoint_name)
        if loaded:
            self.init_calibration_done = True
            self.set_Affine(affine_matrix_left,affine_matrix_right)

    def init_calibrate(self,pupil_points_left,pupil_points_right,init_calibrating_points):
        isDone = True
        try:
            Affine_M_left  = self.getAffineM(pupil_points_left,  init_calibrating_points.astype(np.float32))
            Affine_M_right = self.getAffineM(pupil_points_right, init_calibrating_points.astype(np.float32))
        except Exception as e:
            isDone = False
            print(f"Initial Calibration error: {e}")
            print("Try init Calibration again")
        if isDone:
            self.set_Affine(Affine_M_left,Affine_M_right)
            self.save_ckpt(self.checkpoint_name,Affine_M_left,Affine_M_right)
            self.init_calibration_done = True
            print("Affine Left:",Affine_M_left)
            print("Affine Right:",Affine_M_right)

        return isDone

    def incr_calibrate(self):
        pass

    def getAffineM(self,src,dst):
        A = []
        B = []
        for (x, y), (x_prime, y_prime) in zip(src, dst):
            A.append([x, y, 1, 0, 0, 0])
            A.append([0, 0, 0, x, y, 1])
            B.append(x_prime)
            B.append(y_prime)

        A = np.array(A)
        B = np.array(B)
        affine_params = np.linalg.solve(A, B)
        M = affine_params.reshape(2, 3)
        return M

    def applyAffine(self,src,M):
        dst = np.concatenate((src.reshape((1,2)),np.ones((1,1))),axis=1)
        dst = dst.astype(np.float32)
        dst = np.dot(dst,M.T)
        return dst[0,0:2]

    def transform(self,src_left,src_right):
        dst_left  = self.applyAffine(src_left, self.Affine_M_left)
        dst_right = self.applyAffine(src_right,self.Affine_M_right)
        dst       = (dst_left + dst_right) / 2
        return dst 

class transformation_perspective(transformation_base):
    def load_ckpt(self,filename):
        if os.path.exists(filename):
            try:
                data = np.loadtxt(filename)
                if data.shape == (3, 6):
                    matrix1 = data[:, :3]
                    matrix2 = data[:, 3:]
                    return True, np.array(matrix1), np.array(matrix2)
                else:
                    print("Calibration Checkpoint file: Incorrect format!")
            except Exception as e:
                print(f"Calibration Checkpoint reading error: {e}")
        else:
            print("Calibration Checkpoint does not exists, will perform initial calibration first")
        return False, None, None

    def save_ckpt(self,filename, matrix1, matrix2):
        if (isinstance(matrix1, np.ndarray) and matrix1.shape == (3, 3) and
        isinstance(matrix2, np.ndarray) and matrix2.shape == (3, 3)):
            try:
                combined_matrix = np.hstack((matrix1, matrix2))
                np.savetxt(filename, combined_matrix)
                print("Saved to Calibration Checkpoint")
            except Exception as e:
                print(f"Calibration Checkpoint writting error: {e}")
        else:
            print("Matrix data: Incorrect format!")

    def set_Perspective(self,Perspective_M_left,Perspective_M_right):
        self.Perspective_M_left  = Perspective_M_left 
        self.Perspective_M_right = Perspective_M_right
    
    def __init__(self):
        self.init_calibration_frame_count = 90
        self.init_calibrating_pins        = [[0.9,0.1],[0.1,0.1],[0.1,0.9],[0.9,0.9]]
        self.init_calibration_done        = False
        self.checkpoint_name              = "Cali_ckpt_perspective.txt"

        # Check if checkpoint exists (if exists can skip init calibration)
        self.init_calibration_done = False
        # Transformation
        pts = np.float32([[0,0],[0,1],[1,0],[1,1]])
        self.set_Perspective(cv2.getPerspectiveTransform(pts, pts),cv2.getPerspectiveTransform(pts, pts))

        loaded, perspective_matrix_left,perspective_matrix_right  = self.load_ckpt(self.checkpoint_name)
        if loaded:
            self.init_calibration_done = True
            self.set_Perspective(perspective_matrix_left,perspective_matrix_right)

    def init_calibrate(self,pupil_points_left,pupil_points_right,init_calibrating_points):
        isDone = True
        try:
            Perspective_M_left  = self.getPerspectiveM(pupil_points_left,  init_calibrating_points.astype(np.float32))
            Perspective_M_right = self.getPerspectiveM(pupil_points_right, init_calibrating_points.astype(np.float32))
        except Exception as e:
            isDone = False
            print(f"Initial Calibration error: {e}")
            print("Try init Calibration again")
        if isDone:
            self.set_Perspective(Perspective_M_left,Perspective_M_right)
            self.save_ckpt(self.checkpoint_name,Perspective_M_left,Perspective_M_right)
            self.init_calibration_done = True
            print("Perspective Left:",Perspective_M_left)
            print("Perspective Right:",Perspective_M_right)

        return isDone

    def update_perspective_matrix(self,M, src_points, dst_points, learning_rate=0.001, iterations=1000):
        src_points_homogeneous = np.hstack((src_points, np.ones((src_points.shape[0], 1))))

        for _ in range(iterations):
            projected_points = src_points_homogeneous @ M.T
            projected_points /= projected_points[:, [2]]

            error = projected_points[:, :2] - dst_points
            loss = np.sum(error ** 2)

            gradient = np.zeros_like(M)
            for i in range(src_points.shape[0]):
                x_src, y_src, _ = src_points_homogeneous[i]
                x_proj, y_proj, w_proj = projected_points[i]
                x_dst, y_dst = dst_points[i]

                dx = x_proj - x_dst
                dy = y_proj - y_dst

                gradient[0, 0] += 2 * dx * x_src / w_proj
                gradient[0, 1] += 2 * dx * y_src / w_proj
                gradient[0, 2] += 2 * dx / w_proj

                gradient[1, 0] += 2 * dy * x_src / w_proj
                gradient[1, 1] += 2 * dy * y_src / w_proj
                gradient[1, 2] += 2 * dy / w_proj

                gradient[2, 0] += 2 * (dx * x_src * (-x_proj / (w_proj ** 2)) + dy * x_src * (-y_proj / (w_proj ** 2)))
                gradient[2, 1] += 2 * (dx * y_src * (-x_proj / (w_proj ** 2)) + dy * y_src * (-y_proj / (w_proj ** 2)))
                gradient[2, 2] += 2 * (dx * (-x_proj / (w_proj ** 2)) + dy * (-y_proj / (w_proj ** 2)))

            M -= learning_rate * gradient

        return M
    
    def incr_calibrate(self):
        pass

    # def getPerspectiveM(self,src,dst):
    #     A = []
    #     for i in range(4):
    #         x, y = src[i][1], src[i][0]
    #         x_prime, y_prime = dst[i][1], dst[i][0]
            
    #         A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
    #         A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
        
    #     A = np.array(A)
        
    #     U, S, Vh = np.linalg.svd(A)
    #     L = Vh[-1, :] / Vh[-1, -1]
        
    #     M = L.reshape(3, 3)
    #     return M

    def getPerspectiveM(self,src,dst):
        return cv2.getPerspectiveTransform(src,dst)

    def applyPerspective(self,src,M):
        dst = np.concatenate((src.reshape((1,2)),np.ones((1,1))),axis=1)
        dst = dst.astype(np.float32)
        dst = np.dot(dst,M.T)
        return dst[0,0:2] / dst[0,2]

    def transform(self,src_left,src_right):
        dst_left  = self.applyPerspective(src_left, self.Perspective_M_left)
        dst_right = self.applyPerspective(src_right,self.Perspective_M_right)
        dst       = (dst_left + dst_right) / 2
        return dst 

