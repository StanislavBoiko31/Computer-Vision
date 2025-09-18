from graphics import *
import numpy as np
import time
import math as mt

def task1():
    win_size = 600
    win = GraphWin("2D ромб: обертання, переміщення, масштабування", win_size, win_size)
    win.setBackground('white')

    center = np.array([win_size // 2, win_size // 2])
    diag_v = 140
    diag_h = 80


    diamond = np.array([
        [center[0], center[1] - diag_v // 2, 1],
        [center[0] + diag_h // 2, center[1], 1],
        [center[0], center[1] + diag_v // 2, 1],
        [center[0] - diag_h // 2, center[1], 1]
    ])

    def rotate(points, angle_deg, center):
        angle_rad = np.deg2rad(angle_deg)
        cx, cy = center
        T1 = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0, 1]])
        R = np.array([[mt.cos(angle_rad), -mt.sin(angle_rad), 0],
                      [mt.sin(angle_rad),  mt.cos(angle_rad), 0],
                      [0, 0, 1]])
        T2 = np.array([[1, 0, cx],
                       [0, 1, cy],
                       [0, 0, 1]])
        return points @ T1.T @ R.T @ T2.T

    def translate(points, dx, dy):
        T = np.array([[1, 0, dx],
                      [0, 1, dy],
                      [0, 0, 1]])
        return points @ T.T



    def scale(points, sx, sy, center):
        cx, cy = center
        T1 = np.array([[1, 0, -cx],
                       [0, 1, -cy],
                       [0, 0, 1]])
        S = np.array([[sx, 0, 0],
                      [0, sy, 0],
                      [0, 0, 1]])
        T2 = np.array([[1, 0, cx],
                       [0, 1, cy],
                       [0, 0, 1]])
        return points @ T1.T @ S.T @ T2.T

    steps = 60
    dx, dy = 4, 2
    angle_step = 6
    scale_step = 0.05
    current_diamond = diamond.copy()
    trajectory_color = 'lightblue'

    for i in range(steps):
        factor = 1 + scale_step * mt.sin(i * 2 * mt.pi / steps)
        current_diamond = rotate(current_diamond, angle_step, center)
        current_diamond = translate(current_diamond, dx, dy)
        center += np.array([dx, dy])
        current_diamond = scale(current_diamond, factor, factor, center)
        
        pts = [Point(*current_diamond[j, :2]) for j in range(4)]
        poly = Polygon(*pts)
        poly.setOutline(trajectory_color)
        poly.setFill('yellow')
        poly.draw(win)
        time.sleep(0.07)

    pts = [Point(*current_diamond[j, :2]) for j in range(4)]
    poly = Polygon(*pts)
    poly.setOutline('red')
    poly.setFill('orange')
    poly.draw(win)

    win.getMouse()
    win.close()

def task2():
    win_size = 600
    win = GraphWin("3D Паралелепіпед", win_size, win_size)
    win.setBackground('white')

    size_x, size_y, size_z = 120, 80, 60
    center = np.array([win_size // 2, win_size // 2, win_size // 2])
    cube = np.array([
        [-size_x/2, -size_y/2, -size_z/2, 1],
        [ size_x/2, -size_y/2, -size_z/2, 1],
        [ size_x/2,  size_y/2, -size_z/2, 1],
        [-size_x/2,  size_y/2, -size_z/2, 1],
        [-size_x/2, -size_y/2,  size_z/2, 1],
        [ size_x/2, -size_y/2,  size_z/2, 1],
        [ size_x/2,  size_y/2,  size_z/2, 1],
        [-size_x/2,  size_y/2,  size_z/2, 1],
    ])

    def translate(points, dx, dy, dz):
        T = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])
        return points @ T.T

    def rotate(points, angle_deg, axis='y'):
        angle_rad = np.deg2rad(angle_deg)
        if axis == 'x':
            R = np.array([[1, 0, 0, 0],
                          [0, mt.cos(angle_rad), -mt.sin(angle_rad), 0],
                          [0, mt.sin(angle_rad),  mt.cos(angle_rad), 0],
                          [0, 0, 0, 1]])
        elif axis == 'y':
            R = np.array([[mt.cos(angle_rad), 0, mt.sin(angle_rad), 0],
                          [0, 1, 0, 0],
                          [-mt.sin(angle_rad), 0, mt.cos(angle_rad), 0],
                          [0, 0, 0, 1]])
        elif axis == 'z':
            R = np.array([[mt.cos(angle_rad), -mt.sin(angle_rad), 0, 0],
                          [mt.sin(angle_rad),  mt.cos(angle_rad), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        else:
            R = np.eye(4)
        return points @ R.T

    def axonometric(points, phi_deg=45, theta_deg=30):
        points = rotate(points, phi_deg, axis='y')
        points = rotate(points, theta_deg, axis='x')
        return points

    def project(points):
        P = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 1]])
        return points @ P.T

    def to_screen(points):
        return [Point(points[i,0], points[i,1]) for i in range(points.shape[0])]

    faces = [
        [0,1,2,3],
        [4,5,6,7],
        [0,1,5,4],
        [2,3,7,6],
        [1,2,6,5],
        [0,3,7,4],
    ]
    
    edges = [
        [0,1], [1,2], [2,3], [3,0],
        [4,5], [5,6], [6,7], [7,4],
        [0,4], [1,5], [2,6], [3,7]
    ]

    steps = 60
    colors = ['yellow', 'orange', 'lightgreen', 'lightblue', 'violet', 'pink', 'cyan', 'red']

    drawn_objects = []
    
    for i in range(steps):
        for obj in drawn_objects:
            obj.undraw()
        drawn_objects.clear()
        
        alpha = i / steps
        fill_color = colors[i % len(colors)]
        angle = 360 * alpha
        
        brightness = abs(mt.sin(alpha * 2 * mt.pi))
        
        obj = cube.copy()
        obj = axonometric(obj, phi_deg=35, theta_deg=25)
        obj = rotate(obj, angle, axis='y')
        obj = translate(obj, win_size//2, win_size//2, 0)
        obj2d = project(obj)

        for face in faces:
            pts = to_screen(obj2d[face])
            poly = Polygon(*pts)
            
            base_color = colors[i % len(colors)]
            if brightness > 0.9:
                bright_color = 'white'
            elif brightness < 0.1:
                bright_color = 'black'
            elif base_color == 'yellow':
                bright_color = 'yellow' if brightness > 0.5 else 'gold'
            elif base_color == 'orange':
                bright_color = 'orange' if brightness > 0.5 else 'darkorange'
            elif base_color == 'lightgreen':
                bright_color = 'lightgreen' if brightness > 0.5 else 'darkgreen'
            elif base_color == 'lightblue':
                bright_color = 'lightblue' if brightness > 0.5 else 'darkblue'
            elif base_color == 'violet':
                bright_color = 'violet' if brightness > 0.5 else 'darkviolet'
            elif base_color == 'pink':
                bright_color = 'pink' if brightness > 0.5 else 'deeppink'
            elif base_color == 'cyan':
                bright_color = 'cyan' if brightness > 0.5 else 'darkcyan'
            else:
                bright_color = 'red' if brightness > 0.5 else 'darkred'
            
            poly.setFill(bright_color)
            poly.setOutline('black')
            poly.draw(win)
            drawn_objects.append(poly)
            
        for edge in edges:
            pts = to_screen(obj2d[edge])
            line = Line(pts[0], pts[1])
            if brightness > 0.9:
                edge_color = 'white'
            elif brightness < 0.1:
                edge_color = 'black'
            else:
                edge_color = 'blue' if brightness > 0.5 else 'navy'
            line.setOutline(edge_color)
            line.setWidth(2)
            line.draw(win)
            drawn_objects.append(line)
            
        time.sleep(0.07)

    win.getMouse()
    win.close()

if __name__ == "__main__":
    task1()
    task2()