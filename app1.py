import streamlit as st
import numpy as np
import plotly.graph_objects as go

# make page wide so the plot has plenty of room
st.set_page_config(layout="wide")

# ---- geometry helpers ------------------------------------------------------
class Object3D:
    def __init__(self, vertices, edges=None, draw_points=False):
        self.vertices = vertices
        self.edges = edges or []
        self.draw_points = draw_points

def create_sphere_wireframe(radius=1.5, resolution=20):
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    vertices = []
    edges = []
    index = 0
    # latitude circles
    for i, phi in enumerate(v):
        for theta in u:
            x = radius * np.cos(theta) * np.sin(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(phi) + 8    # move forward like cube
            vertices.append((x, y, z))
            if i > 0:
                edges.append((index, index - resolution))
            if theta != u[-1]:
                edges.append((index, index + 1))
            index += 1

    # longitude circles
    for theta in u:
        for phi in v:
            x = radius * np.cos(theta) * np.sin(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(phi) + 8
            vertices.append((x, y, z))
            if phi != v[-1]:
                edges.append((index, index + 1))
            index += 1

    return Object3D(np.array(vertices), edges)

def generate_sphere_points(radius=1.5, resolution=30):
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    pts[:, 2] += 8
    return pts

def get_rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    R_y = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [ 0,           1, 0          ],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    R_x = np.array([
        [1, 0, 0],
        [0,  np.cos(pitch), -np.sin(pitch)],
        [0,  np.sin(pitch),  np.cos(pitch)]
    ])
    R_z = np.array([
        [ np.cos(roll), -np.sin(roll), 0],
        [ np.sin(roll),  np.cos(roll), 0],
        [ 0, 0, 1]
    ])
    return R_z @ R_x @ R_y

def project_points(points, P, perspective=True):
    ones = np.ones((points.shape[0], 1))
    ph = np.hstack((points, ones)).T        # 4×N
    proj = P @ ph                          # 3×N
    X, Y, W = proj
    if perspective:
        x_proj = X / W
        y_proj = Y / W
    else:
        x_proj = X
        y_proj = Y
    return np.vstack((x_proj, y_proj)).T

# cross‑ratio line in 3‑D
line_points = np.array([
    [-3, -1, 5],  # A
    [-1, -1, 5],  # B
    [ 2, -1, 5],  # C
    [ 4, -1, 5]   # D
])

def cross_ratio(A, B, C, D):
    return ((A - C) * (B - D)) / ((A - D) * (B - C))

# scene objects (copied verbatim from geonew1.py)
cube_pts = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                     [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
cube_pts[:,2] += 8
cube_edges = [(0,1),(1,2),(2,3),(3,0),
              (4,5),(5,6),(6,7),(7,4),
              (0,4),(1,5),(2,6),(3,7)]
cube = Object3D(cube_pts, cube_edges)

pyr_pts = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[0,0,1]])
pyr_pts[:,2] += 8
pyramid = Object3D(pyr_pts,
                   [(0,1),(1,2),(2,3),(3,0),
                    (0,4),(1,4),(2,4),(3,4)])

plane = Object3D(np.array([[-3,0,5],[3,0,5],[3,0,11],[-3,0,11]]),
                 [(0,1),(1,2),(2,3),(3,0)])

cloud = np.random.uniform(-2,2,(60,3))
cloud[:,2] += 8
pointcloud = Object3D(cloud, draw_points=True)

objects = {"Cube": cube,
           "Pyramid": pyramid,
           "Plane": plane,
           "Point Cloud": pointcloud}

# ---------------------------------------------------------------------------
def main():
    st.title("Projective‑geometry 3‑D viewer")

    # sidebar controls (borrowed layout from the previous app)
    with st.sidebar:
        choice = st.selectbox("Object", list(objects.keys()) + ["Sphere"])
        sphere_mode = st.radio("Sphere mode", ("Wireframe","Points"))

        # instead of this:
        # yaw = st.slider("Yaw", -70.0, 70.0, 20.0)
        # pitch = st.slider("Pitch", -70.0, 70.0, 10.0)
        # …
        # use number_input boxes:

        yaw   = st.number_input("Yaw",   min_value=-180.0, max_value=180.0,
                                    value=20.0, step=0.1, format="%.1f")
        pitch = st.number_input("Pitch", min_value=-180.0, max_value=180.0,
                                    value=10.0, step=0.1, format="%.1f")
        roll  = st.number_input("Roll",  min_value=-180.0, max_value=180.0,
                                    value=0.0,  step=0.1, format="%.1f")

        tx = st.number_input("Cam X", min_value=-10.0, max_value=10.0,
                             value=0.0, step=0.1, format="%.1f")
        ty = st.number_input("Cam Y", min_value=-10.0, max_value=10.0,
                             value=0.0, step=0.1, format="%.1f")
        tz = st.number_input("Cam Z", min_value=-10.0, max_value=10.0,
                             value=0.0, step=0.1, format="%.1f")

        f  = st.number_input("Focal length", min_value=0.1, max_value=50.0,
                             value=5.0, step=0.1, format="%.1f")
        cx = st.number_input("Principal X", min_value=-10.0, max_value=10.0,
                             value=0.0, step=0.1, format="%.1f")
        cy = st.number_input("Principal Y", min_value=-10.0, max_value=10.0,
                             value=0.0, step=0.1, format="%.1f")

        persp = st.checkbox("Perspective", value=True)
        show_inf = st.checkbox("Line at ∞", value=False)

    # pick object (including the extra sphere modes)
    if choice == "Sphere":
        if sphere_mode == "Wireframe":
            obj = create_sphere_wireframe()
        else:
            obj = Object3D(generate_sphere_points(), draw_points=True)
    else:
        obj = objects[choice]

    # camera matrices as in geonew1.update()
    C = np.array([tx, ty, tz])
    R = get_rotation_matrix(yaw, pitch, roll)
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    Rt = np.hstack((R.T, -R.T @ C.reshape(3,1)))
    P = K @ Rt
    KR = K @ R   # used for affine display

    coords2d = project_points(obj.vertices, P, perspective=persp)

    fig = go.Figure()

    # draw object (no legend entries)
    if obj.draw_points:
        fig.add_trace(go.Scatter(x=coords2d[:,0], y=coords2d[:,1],
                                 mode="markers", marker=dict(size=3),
                                 showlegend=False))
    else:
        for a,b in obj.edges:
            fig.add_trace(go.Scatter(x=[coords2d[a,0], coords2d[b,0]],
                                     y=[coords2d[a,1], coords2d[b,1]],
                                     mode="lines", line=dict(color="black"),
                                     showlegend=False))

    # vanishing points / convergence lines
    vp_coords = {}
    if persp:
        vx, vy, vz = P[:,0], P[:,1], P[:,2]
        for color, v in (('red',vx), ('green',vy), ('blue',vz)):
            Xv, Yv, Wv = v
            # keep homogeneous triple for horizon computation
            vp_coords[color] = np.array([Xv, Yv, Wv])
            if abs(Wv) > 1e-6:                    # finite vp
                vp_x, vp_y = Xv/Wv, Yv/Wv
                fig.add_trace(go.Scatter(
                    x=[vp_x], y=[vp_y],
                    mode="markers",
                    marker=dict(size=8, color=color),
                    name=f"{color} vp"            # legend entry
                ))
            else:                                 # point at infinity
                dir2 = np.array([Xv, Yv])
                n = np.linalg.norm(dir2)
                if n > 1e-6:
                    dir2 /= n
                    fig.add_trace(go.Scatter(
                        x=[0, dir2[0]*20], y=[0, dir2[1]*20],
                        mode="lines",
                        line=dict(color=color, dash='dash'),
                        showlegend=False
                    ))
                    fig.add_annotation(
                        x=dir2[0]*20, y=dir2[1]*20,
                        text=f"{color} ∞",
                        showarrow=False, font=dict(color=color)
                    )

        # convergence/vanishing lines (draw from edge midpoints to vp)
        if choice in ["Cube","Pyramid","Plane"]:
            for edge in obj.edges[::3]:
                p1 = coords2d[edge[0]]
                p2 = coords2d[edge[1]]
                d3 = obj.vertices[edge[1]] - obj.vertices[edge[0]]
                dh = np.append(d3, 0)
                v = P @ dh
                Xv,Yv,Wv = v
                if abs(Wv) > 1e-6:
                    vp_x, vp_y = Xv/Wv, Yv/Wv
                    mid = (p1+p2)/2
                    d3 = d3 / np.linalg.norm(d3)
                    axis = np.argmax(np.abs(d3))
                    color = ['red','green','blue'][axis]
                    fig.add_trace(go.Scatter(
                        x=[mid[0], vp_x], y=[mid[1], vp_y],
                        mode='lines',
                        line=dict(color=color, dash='dot'),
                        opacity=0.6,
                        showlegend=False
                    ))

        # horizon/line‑at‑infinity (handles vertical case too)
        if show_inf and 'red' in vp_coords and 'blue' in vp_coords:
            l = np.cross(vp_coords['red'], vp_coords['blue'])
            a, b, c = l
            if not (abs(a) < 1e-6 and abs(b) < 1e-6):
                if abs(b) > 1e-6:
                    xvals = np.array([-20, 20])
                    yvals = -(a*xvals + c) / b
                else:  # vertical horizon
                    xval = -c/a if abs(a) > 1e-6 else 0
                    xvals = np.array([xval, xval])
                    yvals = np.array([-20, 20])
                fig.add_trace(go.Scatter(
                    x=xvals, y=yvals,
                    mode='lines',
                    line=dict(color='orange', dash='dash'),
                    showlegend=False
                ))
                fig.add_annotation(
                    x=xvals[0], y=yvals[0],
                    text='ℓ∞ (horizon)',
                    showarrow=False, font=dict(color='orange')
                )

    # cross‑ratio demonstration
    proj_line = project_points(line_points, P, perspective=persp)
    fig.add_trace(go.Scatter(x=proj_line[:,0], y=proj_line[:,1],
                             mode="lines", line=dict(color='purple', width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=proj_line[:,0], y=proj_line[:,1],
                             mode="markers+text",
                             text=['A','B','C','D'],
                             textposition='top right',
                             marker=dict(size=8, color='purple'),
                             showlegend=False))
    cr3 = cross_ratio(*line_points[:,0])
    cr2 = cross_ratio(proj_line[0,0], proj_line[1,0],
                      proj_line[2,0], proj_line[3,0])
    st.markdown(f"**Cross‑ratio 3‑D**: {cr3:.3f}  **2‑D**: {cr2:.3f}")

    # matrix display
    if persp:
        P_disp = np.round(P,2)
        st.markdown("**Projective camera matrix P (3×4):**")
        st.text(str(P_disp))
    else:
        KR_disp = np.round(KR,2)
        st.markdown("**Affine linear matrix KR (3×3):**")
        st.text(str(KR_disp))

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=600, margin=dict(l=0,r=0,b=0,t=0))

    # **do not disable legend globally** – vp entries will appear
    # fig.update_layout(showlegend=False)

    st.plotly_chart(fig, width="stretch")
    st.write("Vanishing points:")
    st.write(vx, vy, vz)

if __name__ == "__main__":
    main()