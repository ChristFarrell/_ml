const express = require('express');
const { engine } = require('express-handlebars');
const marked = require('marked');
const { Sequelize, DataTypes } = require('sequelize');
const path = require('path');

const app = express();
const PORT = 3000;

// 設定 Handlebars
app.engine('handlebars', engine({
  defaultLayout: 'main',
  helpers: {
    marked: function(content) {
      return marked(content || '');
    },
    formatDate: function(date) {
      return new Date(date).toLocaleDateString('zh-TW');
    }
  }
}));
app.set('view engine', 'handlebars');
app.set('views', path.join(__dirname, 'views'));

// 中間件
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// 初始化 SQLite 資料庫
const sequelize = new Sequelize({
  dialect: 'sqlite',
  storage: path.join(__dirname, 'database.sqlite'),
  logging: false
});

// 定義 Post 模型
const Post = sequelize.define('Post', {
  title: {
    type: DataTypes.STRING,
    allowNull: false
  },
  content: {
    type: DataTypes.TEXT,
    allowNull: false
  },
  slug: {
    type: DataTypes.STRING,
    unique: true,
    allowNull: false
  },
  published: {
    type: DataTypes.BOOLEAN,
    defaultValue: false
  }
});

// 同步資料庫
sequelize.sync();

// 路由

// 首頁 - 顯示所有已發布的文章
app.get('/', async (req, res) => {
  try {
    const posts = await Post.findAll({
      where: { published: true },
      order: [['createdAt', 'DESC']]
    });
    res.render('home', { posts });
  } catch (error) {
    res.render('error', { error: error.message });
  }
});

// 顯示單篇文章
app.get('/post/:slug', async (req, res) => {
  try {
    const post = await Post.findOne({ where: { slug: req.params.slug } });
    if (!post) {
      return res.status(404).render('404');
    }
    res.render('post', { post });
  } catch (error) {
    res.render('error', { error: error.message });
  }
});

// 管理頁面
app.get('/admin', async (req, res) => {
  try {
    const posts = await Post.findAll({ order: [['createdAt', 'DESC']] });
    res.render('admin', { posts });
  } catch (error) {
    res.render('error', { error: error.message });
  }
});

// 新增文章頁面
app.get('/admin/new', (req, res) => {
  res.render('edit', { post: {}, action: '/admin/new', isNew: true });
});

// 編輯文章頁面
app.get('/admin/edit/:id', async (req, res) => {
  try {
    const post = await Post.findByPk(req.params.id);
    if (!post) {
      return res.status(404).render('404');
    }
    res.render('edit', { post, action: `/admin/edit/${post.id}`, isNew: false });
  } catch (error) {
    res.render('error', { error: error.message });
  }
});

// 建立新文章
app.post('/admin/new', async (req, res) => {
  try {
    const { title, content, slug, published } = req.body;
    await Post.create({ title, content, slug, published: published === 'on' });
    res.redirect('/admin');
  } catch (error) {
    res.render('error', { error: error.message });
  }
});

// 更新文章
app.post('/admin/edit/:id', async (req, res) => {
  try {
    const { title, content, slug, published } = req.body;
    const post = await Post.findByPk(req.params.id);
    if (post) {
      await post.update({ title, content, slug, published: published === 'on' });
    }
    res.redirect('/admin');
  } catch (error) {
    res.render('error', { error: error.message });
  }
});

// 刪除文章
app.post('/admin/delete/:id', async (req, res) => {
  try {
    const post = await Post.findByPk(req.params.id);
    if (post) {
      await post.destroy();
    }
    res.redirect('/admin');
  } catch (error) {
    res.render('error', { error: error.message });
  }
});

// 啟動伺服器
app.listen(PORT, () => {
  console.log(`網誌系統已啟動: http://localhost:${PORT}`);
  console.log(`管理頁面: http://localhost:${PORT}/admin`);
});
