# Set-Valued Hénon Map Visualization

A **WebAssembly-powered** visualization tool for comparing deterministic and set-valued (noisy) Hénon map dynamics. This project demonstrates chaotic behavior in dynamical systems using high-performance Rust computation with an interactive React frontend.


## 🎯 **Project Overview**

This application visualizes the **Hénon Map**, a discrete-time dynamical system that exhibits chaotic behavior:

```
x_{n+1} = 1 - ax_n² + y_n
y_{n+1} = bx_n
```

**Key Features:**
- **Deterministic Hénon Map**: Classic chaotic attractor visualization
- **Set-Valued Hénon Map**: Bounded noise version showing uncertainty quantification
- **Side-by-side comparison**: Real-time visualization of both systems
- **Interactive parameters**: Adjust system parameters and noise levels
- **High-performance rendering**: WebAssembly backend for fast computation

## 🛠️ **Tech Stack**

- **Backend**: Rust + WebAssembly (`wasm-bindgen`)
- **Frontend**: React + Vite
- **Visualization**: HTML5 Canvas
- **Math**: Custom chaotic dynamics implementation
- **Build Tools**: `wasm-pack` for WebAssembly compilation

## 📋 **Prerequisites**

### **Required Software**

1. **Node.js** (v20.19+ or v22.12+)
   ```bash
   # Check your version
   node --version
   
   # Install from: https://nodejs.org/
   ```

2. **Rust** (latest stable)
   ```bash
   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   
   # Verify installation
   rustc --version
   cargo --version
   ```

3. **wasm-pack** (WebAssembly build tool)
   ```bash
   # Install wasm-pack
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   
   # Verify installation
   wasm-pack --version
   ```

### **Additional Rust Targets**

```bash
# Add WebAssembly target for Rust
rustup target add wasm32-unknown-unknown
```

## 🚀 **Getting Started**

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

The application will be available at: **http://localhost:5173** (or next available port)

## 📁 **Project Structure**

```
set-valued-viz/
├── src/                          # Rust source code
│   ├── lib.rs                   # Main WebAssembly exports
│   └── main.rs                  # CLI entry point (unused)
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── App.jsx              # Main React component
│   │   ├── App.css              # Styling
│   │   ├── pkg/                 # WebAssembly files (copied)
│   │   └── ...
│   ├── package.json             # Node.js dependencies
│   └── vite.config.js           # Vite configuration
├── pkg/                         # Generated WebAssembly module
├── Cargo.toml                   # Rust dependencies
└── README.md                    # This file
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

## 🎮 **Using the Application**

### **System Parameters**
- **a**: Hénon parameter (default: 1.4)
- **b**: Hénon parameter (default: 0.3)
- **x0, y0**: Initial conditions (default: 0.1, 0.1)

### **Noise Parameters**
- **εx, εy**: Noise bounds for set-valued version (default: 0.005)

### **Generation Controls**
- **Iterations**: Number of trajectory points to generate
- **Skip Transient**: Skip initial iterations to focus on attractor
- **Generate Trajectories**: Create new visualizations

### **Expected Results**
- **Left Canvas**: Deterministic Hénon attractor (banana-shaped)
- **Right Canvas**: Set-valued version with bounded uncertainty
- **Identical deterministic results** with same parameters
- **Variable noisy results** showing uncertainty bounds

## 🐛 **Troubleshooting**

### **Common Issues**

1. **"wasm-pack not found"**
   ```bash
   # Install wasm-pack
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   ```

2. **"Node.js version not supported"**
   - Update Node.js to v20.19+ or v22.12+
   - Use `nvm` for version management

3. **"WebAssembly module failed to load"**
   ```bash
   # Ensure WASM files are copied correctly
   cd frontend
   npm run build-wasm
   ```

4. **"State values too large" warnings**
   - This is normal behavior for chaotic systems
   - The system automatically resets divergent trajectories
   - Try reducing noise parameters (εx, εy) if excessive

### **Build Issues**

1. **Rust compilation errors**:
   ```bash
   # Update Rust
   rustup update
   
   # Ensure WebAssembly target is installed
   rustup target add wasm32-unknown-unknown
   ```

2. **Frontend build errors**:
   ```bash
   # Clear node modules and reinstall
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

## 📊 **Performance Notes**

- **Iteration Limits**: Maximum 50,000 iterations for performance
- **Memory Management**: WASM objects are automatically cleaned up
- **Browser Requirements**: Modern browsers with WebAssembly support
- **Recommended Settings**: 1000-5000 iterations for smooth performance

## 🧮 **Mathematical Background**

The **Hénon Map** is a discrete-time dynamical system:

```
x_{n+1} = 1 - ax_n² + y_n
y_{n+1} = bx_n
```

**Standard Parameters:**
- `a = 1.4, b = 0.3`: Classic chaotic attractor
- **Attractor bounds**: roughly `x ∈ [-1.5, 1.5]`, `y ∈ [-0.4, 0.4]`

**Set-Valued Extension:**
```
x_{n+1} = 1 - ax_n² + y_n + ξx
y_{n+1} = bx_n + ξy
```
where `ξx, ξy ∈ [-ε, +ε]` represent bounded uncertainty.

**Happy visualizing! 🎉**