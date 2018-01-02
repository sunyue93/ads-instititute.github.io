# How to post a blog

Steps:
1. create a file in the folder _posts
2. the file name should of the form 2017-12-20-start.md
3. the file starts with<br>
```
---
layout:     post
title:      TITLE
date:       2017-12-20 (the date you want to document to be posted)
summary:    One sentence summary
---
your content
```

# How to post a blog

Use the following header instead
```
---
layout:     post
title:      TITLE
date:       2017-12-20 (the date you want to document to be posted)
summary:    One sentence summary
categories: news
---
your content
```

# How to write the content
Please see the instruction [here](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).
For latex, 
Inline mode:
```
Someone really likes $$E=mc^2$$
```

Display mode:
```


$$
x^2
$$

```
Note that the display mode requires one empty line before and after. (Otherwise, there is some conflict with markdown syntax such as the underscore will be interpreted as emphasis. 

# How to add a new member
1. add an entry in _data\people.yml
```
    - first_name: Dmitriy
      last_name: Drusvyatskiy
      pic: FILENAME
      interest: Convex Optimization
      url: http://sites.math.washington.edu/~ddrusv/
```
2. **check the update using http://yaml-online-parser.appspot.com/**
3. add an image in members\FILENAME

# How to add a new publication
Add an entry in _data\publications.yml
```
    - author: Sbastien Bubeck, Michael B. Cohen, James R. Lee, Yin Tat Lee, Aleksander Madry
      title: k-server via multiscale entropic regularization
      journal: arXiv abs/1711.01085
      url: http://arxiv.org/abs/1711.01085
      blog: ./2017/12/20/kserver
      highlight: A breakthough in the k-server problem. The first o(k)-competitive algorithm for hierarchically separated trees.
      code: https://github.com/
```
