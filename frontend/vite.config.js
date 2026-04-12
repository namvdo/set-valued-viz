import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: process.env.VITE_BASE_URL || './',
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.js',
    globals: true
  },
  server: {
    port: 5173,
    host: true,
    fs: {
      allow: ['..']
    }
  },
  build: {
    target: 'esnext'
  },
  optimizeDeps: {
    exclude: ['./pkg/set_valued_viz.js']
  },
  assetsInclude: ['**/*.wasm'],
  resolve: {
    alias: {
      '@pkg': '/pkg'
    }
  }
})
