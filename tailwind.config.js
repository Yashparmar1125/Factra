/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./frontend/templates/**/*.html",
    "./extension/**/*.html",
    "./frontend/static/**/*.js",
    "./extension/**/*.js"
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        factra: {
          900: '#0B0F19', // Deep slate background
          800: '#111827', // Surfaces/Cards
          750: '#172033', // Lighter hover states
          700: '#1F2937', // Borders
        },
        primary: '#3B82F6', // Cyber Blue
        success: '#10B981', // Neon Green (Real)
        danger: '#F43F5E', // Neon Rose (Fake)
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
