/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './templates/**/*.html', // Adjust the paths based on your Flask project structure
    './static/js/**/*.js', // Include JavaScript files if necessary
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}

