import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// Built once into the Python package's serve directory. Vue itself is not
// bundled: trame's client already loads it as the global `Vue`.
export default defineConfig({
  plugins: [vue()],
  define: { "process.env.NODE_ENV": JSON.stringify(process.env.AK_DEV ? "development" : "production") },
  build: {
    outDir: "../trame_flowsheet/serve",
    emptyOutDir: true,
    lib: {
      entry: "src/index.js",
      name: "aquakin_flowsheet",
      formats: ["umd"],
      fileName: () => "aquakin-flowsheet.umd.js",
      cssFileName: "aquakin-flowsheet",
    },
    rollupOptions: {
      external: ["vue"],
      output: { globals: { vue: "Vue" } },
    },
  },
});
