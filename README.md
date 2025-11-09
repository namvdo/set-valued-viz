# Set-Valued Hénon Map Visualization

A **WebAssembly-powered** visualization tool for comparing deterministic and set-valued (noisy) Hénon map dynamics. This project demonstrates chaotic behavior in dynamical systems using high-performance Rust computation with an interactive React frontend.


**Project Overview**

This application visualizes the **Hénon Map**, a discrete-time dynamical system that exhibits chaotic behavior:

```
x_{n+1} = 1 - ax_n² + y_n
y_{n+1} = bx_n
```
Bounded-noise Henon-map:
```
x_{n+1} = 1 - a*x_n^2 + y + ξ
y_{n+1} = b*x_n + η
Where (ξ, η) ∈ B_ε(0) = {(u,v) : √(u² + v²) ≤ ε}
```

**Key Features:**
- **Deterministic Hénon Map**: Classic chaotic attractor visualization
- **Set-Valued Hénon Map**: Bounded noise version showing uncertainty quantification
- **Side-by-side comparison**: Real-time visualization of both systems
- **Interactive parameters**: Adjust system parameters and noise levels
- **High-performance rendering**: WebAssembly backend for fast computation

## **Getting Started**

### **1. Clone the Repository**

```bash
git clone <repository-url>
cd set-valued-viz
```

### **2. Build WebAssembly Module**

```bash
# Build the Rust code to WebAssembly
wasm-pack build --target web --out-dir pkg
```

This creates the WebAssembly module in the `pkg/` directory.

### **3. Install Frontend Dependencies**

```bash
cd frontend
npm install
```

### **4. Copy WASM Files to Frontend**

```bash
# Copy WebAssembly files to the frontend source
npm run build-wasm
```

**Alternative:** Manual copy if the npm script doesn't work:
```bash
cp -r ../pkg/* src/pkg/
```

### **5. Start Development Server**

```bash
npm run dev
```

## 🔧 **Development Workflow**

### **Making Changes to Rust Code**

1. **Edit Rust files** in `src/`
2. **Rebuild WebAssembly**:
   ```bash
   wasm-pack build --target web --out-dir pkg
   ```
3. **Copy to frontend**:
   ```bash
   cd frontend
   npm run build-wasm
   ```
4. **Frontend auto-reloads** (if dev server is running)

### **Making Changes to Frontend**

1. **Edit React files** in `frontend/src/`
2. **Vite auto-reloads** the browser automatically

### **Convenient Build Script**

The frontend includes a convenient build script:

```bash
cd frontend
npm run build-wasm  # Builds Rust → WASM → copies to frontend
```