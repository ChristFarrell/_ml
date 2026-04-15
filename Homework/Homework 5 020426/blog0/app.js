const express = require('express');
const Database = require('better-sqlite3');
const path = require('path');

const app = express();
const db = new Database('blog.db');

// Initialize database
db.exec(`
  CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

// Middleware
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

// Routes
app.get('/', (req, res) => {
  const posts = db.prepare('SELECT * FROM posts ORDER BY created_at DESC').all();
  res.render('index', { posts });
});

app.get('/new', (req, res) => {
  res.render('new');
});

app.post('/posts', (req, res) => {
  const { title, content } = req.body;
  db.prepare('INSERT INTO posts (title, content) VALUES (?, ?)').run(title, content);
  res.redirect('/');
});

const PORT = 3050;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
