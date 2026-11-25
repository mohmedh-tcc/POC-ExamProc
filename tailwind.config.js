/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        proctor: {
          dark: '#0f172a',
          card: '#1e293b',
          border: '#334155',
          accent: '#38bdf8',
          green: '#22c55e',
          amber: '#f59e0b',
          red: '#ef4444',
        }
      }
    },
  },
  plugins: [],
}
