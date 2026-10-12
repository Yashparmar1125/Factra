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
          900: '#070A11', // Darker richer slate
          800: '#111827', // Surfaces
          750: '#172033', // Hover
          700: '#1F2937', // Borders
        },
        primary: '#3B82F6',
        success: '#10B981',
        danger: '#F43F5E',
        accent: '#8B5CF6' // Purple accent for gradients
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
