---
# ============================== NOTE: ============================= #
# Template for SMLab talks - Light Theme
# Template editors: Adhilsha A., Jyotirmaya Shivottam
# License: MIT, ©️ SMLab, NISER Bhubaneswar, 2024
# ------------------------------------------------------------------ #
# Comments are denoted by 'NOTE:' or 'CHANGEME:' in the code.
# Instructions:
# - Install the "Marp for VS Code" extension in VS Code for exporting .pdf.
# - All comments you leave in the slides show up as hints in the presentation mode (if using .pptx or .html).
# - See `./exported-templates/` for the exported files (.pdf, .pptx, .html).
#
# Resources:
# - https://marpit.marp.app/directives - Help on marp styling
# - https://marpit.marp.app/image-syntax - Help on image syntax
# - https://konbert.com/generator/markdown - A good markdown table generator
# - https://mermaid.live/edit - Help on mermaid diagrams (flowcharts, sequence diagrams etc.) - Marp does not support mermaid diagrams directly. Export and insert.
# - https://draw.io - For diagrams
# - https://g.co/kgs/utCPhHg - For color codes
#
# Font Family options: NOTE: You will need to have the font you use, installed in your system. Check ./fonts/ for some of these fonts. Other sources:
# - https://fonts.google.com/ - Google fonts
# - https://github.com/cormullion/juliamono - JuliaMono
# - https://www.nerdfonts.com/font-downloads - Nerd Fonts
# ============================== XXXXX ============================= #

marp: true
math: katex
theme: gaia
paginate: true
backgroundColor: #f5f5f5
color: black
footer: "SMLab Talks | Smith, J. | Short Talk Title | Aug 15, 2024" # CHANGEME: ShortNames, dates
size: 16:9 # (16:9, 4:3 etc.)
style: |
  :root {
    font-size: 27px;
    font-family: Kreon, sans-serif;
    line-height: 1.3; /* NOTE: This vertically spaces the text */
    letter-spacing: 1.2px; /* NOTE: This spaces the letters */
  }
  /* Heading Color */
  h1 {
    color: #1F316F;
  }
  /* Link Color */
  a {
    color: #360066; 
  }
  /* Styles for the Presenter and Lab names */
  .presenter-name {
    text-align: center;
    font-size: 40px;
    color: #5f1fc3;
  }
  .lab-name {
    text-align: center;
    font-size: 30px;
    color: #b33422;
  }
  /* List Color */
  p {
    text-align: justify; 
  }
  /* Footer */
  footer {
    font-size: 20px;
    color: #aa3399;
  }
  /* Pagination */
  section::after {
    content: attr(data-marpit-pagination) '/' attr(data-marpit-pagination-total);
    color: #d9b36f;
  }
  /* CHANGEME: Add more CSS styles here as needed */
---

<!-- NOTE: LEAVE THIS SLIDE AS-IS -->
<!-- _class: lead -->
<!-- _footer: '' -->
<!-- _paginate: false -->
<!-- _backgroundColor: #000 -->
<!-- _backgroundImage: url('./media/slide_bg_light.png') -->
<!-- NOTE: `scoped` is used to limit the styles to the current slide -->
<style scoped>
  h1 {
    margin-top: 75px;
    font-size: 70px;
    color: #f35422;
  }
  .socials {
    margin-top: 30px;
    width: 100%;
    text-align: center;
    vertical-align: middle;
    color: red;
  }
  .socials a {
    text-align: center;
  }
  .socials a img {
    width: 60px;
    height: 60px;
  }
  .logos {
    margin-top: 100px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    text-align: center;
  }
  .logos img {
    width: 200px;
    height: 200px;
    background: white;
    border-radius: 50%;
  }
  a {
    margin: 15px;
    font-size: 40px;
    color: #fff;
    text-decoration: none;
  }
</style>

<!-- Title -->
<h1> Subhankar Mishra Lab Weekly Talks </h1>

<!-- Socials -->
<span class="socials">
<!-- Lab-talks website -->
  <a href="https://smlab.niser.ac.in/labtalks/" target="_blank">
    <img src="./media/web.png" alt="Website">
  </a>
  <!-- Lab-talks GitHub -->
  <a href="https://github.com/JeS24/smlab-talks/" target="_blank">
    <img src="./media/gh.svg" alt="GitHub">
  </a>
  <!-- Lab Twitter -->
  <a href="https://twitter.com/mishra_lab" target="_blank">
    <img src="./media/twitter.svg" alt="Twitter">
  </a>
  <!-- Lab public email -->
  <a href="mailto:smlab@niser.ac.in">
    <img src="./media/email.png" alt="Mail">
  </a>
</span>

<!-- Logos -->
<span class="logos">
  <img src="./media/dae_logo.svg" alt="DAE">
  <img src="./media/hbni_logo.png" alt="HBNI">
  <img src="./media/niser_logo_blue.svg" alt="NISER">
  <img src="./media/smlab_logo.svg" alt="SMLab">
</span>

---

<!-- NOTE: A `_` prefix is used to change properties for only the current slide -->

<!-- _class: lead -->
<!-- _footer: '' -->
<!-- _paginate: false -->
<!-- _backgroundColor: #000 -->
<!-- _backgroundImage: url('./media/slide_bg_light.png') -->

![bg right:60% w:75% opacity:55%](./media/bg2.png)

<!-- NOTE: Multiple images added here - one in BG, one in FG -->

# **Title Slide Variation 1**

<p class="presenter-name">
  John Smith
</p>
<p class="lab-name">
  Subhankar Mishra Lab<br>
  Aug 15, 2024
</p>

---

<!-- NOTE: Another example of title slide -->

<!-- _class: lead -->
<!-- _footer: '' -->
<!-- _paginate: false -->
<!-- _backgroundColor: #000 -->
<!-- _backgroundImage: url('./media/slide_bg_light.png') -->

![bg w:60% opacity:20%](./media/bg2.png)

# **TITLE SLIDE VARIATION 2**

<p class="presenter-name">
  John Smith
</p>
<p class="lab-name">
  Subhankar Mishra Lab<br>
  Aug 15, 2024
</p>

---

# Outline / Table of Contents

<!-- NOTE: In VS Code, you can use extensions like "Markdown All in One" to auto-create Table of Contents with links  -->
<style scoped>
  h1 {
    color: orange;
  }
</style>

- Text
- Bullet Points
- Code
- Equations
- Tables
- Image Slides
- Split Slides
- References
- [_Post-talk Questions_](#post-talk-questions)
<!-- NOTE: This is a example link to other slides -->

---

# Text

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus aliquet odio augue, ac maximus dolor molestie fermentum. Nulla eleifend, lectus in pretium volutpat, mauris mi bibendum eros, sed fringilla nulla lectus sit amet nunc. Vivamus velit quam, sollicitudin eget nisl vel, dignissim tempus metus. Nunc blandit tempus tellus non eleifend. Etiam vulputate, justo eu faucibus blandit, ex dolor hendrerit nulla, eu ullamcorper magna risus a mauris. Cras nec tristique metus, sed consequat libero. Nam tincidunt fringilla ligula ut fringilla. Aliquam nec faucibus felis, et commodo ex. Aliquam leo dui, feugiat quis mollis in, congue quis magna. Cras tempus leo a risus sollicitudin, et hendrerit magna blandit. Suspendisse quis leo mattis, sagittis turpis sit amet, accumsan nulla. Fusce at dui non quam euismod varius eu sed arcu.

---

# Bullet Points

- Point 1
- Point 2
  - Subpoint 1
  - Subpoint 2
    - Sub-subpoint 1
      - Sub-sub-subpoint 1
  - Subpoint 3
- Point 3
  - Subpoint 1

---

# Code

- Here's some code:

  ```py
  def hello_world():
      print("Hello, World!")
  ```

- Here's some inline code: `print("Hello, World!")`
- Here's some code in a different language:

  ```julia
  function hello_world()
    println("Hello, World!")
  end
  ```

---

# Equations

- Adjacency matrix:
  $$
  A_{ij} = \begin{cases}
    1, \quad & \text{if } i \text{ and } j \text{ are connected} \\
    0, \quad & \text{otherwise}
  \end{cases}
  $$
- Adjacency matrix for a SRG satisfies:
  $$
  \begin{align*}
  A^2 &= (k - \mu)I + \mu J + (\lambda - \mu)A \\
  JA &= AJ = kA \\
  J^2 &= NJ
  \end{align*}
  $$
- A graph with $N$ vertices has an $N\times N$ adjacency matrix. This problem is in the $NP$ complexity class and is purported to be NP-Intermediate (if $P \ne NP$).

---

# Tables

<style scoped>
  table {
    /* NOTE: To make the table span the slide width */
    width: 100%;
    font-size: 24px;
  }
</style>

| Heading 1 | Heading 2 | Heading 3 |
| :-------- | :-------: | --------: |
| abc       |    cde    |       fgh |
| 123       |    456    |       789 |

---

# Image Slide 1

<style scoped>
  img {
    border-radius: 50px;
  }
</style>

<!-- NOTE: Setting a background image of a particular width and height -->

![w:1100 h:250 opacity:1.0](./media/img1.jpg)

---

<!-- NOTE: Another example of image as a background -->
<!-- _paginate: false -->
<!-- _footer: '' -->
<style scoped>
  h1 {
    color: black;
  }
</style>

# Image Slide 2

![bg ](./media/img2.jpg)

---

<!-- NOTE: Another example of image as a background -->
<!-- _paginate: false -->
<!-- _footer: '' -->
<style scoped>
  h1 {
    color: black;
  }
</style>

# Image Slide 3

![bg](./media/img2.jpg)
![bg](./media/img2.jpg)

---

<!-- NOTE: Another example of image as a background -->

# Image Slide 4

![bg w:100%](./media/img2.jpg)
![bg w:100%](./media/img2.jpg)

---

# Split Slide 1

![bg right:45% w:90%](./media/img2.jpg)

- First point made here
- Second point made here
- Third point made here

---

# Split Slide 2

![bg right](./media/img2.jpg)

- First point made here
- Second point made here
- Third point made here

---

# References - 1

- [Link to GitHub repo for the code](https://github.com/JeS24/CTQW-graph-isomorphism)
- Main papers:
  1. Rudinger et al; [Noninteracting multiparticle quantum random walks applied to the graph isomorphism problem for strongly regular graphs](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.86.022334)
  2. Gamble et al; [Two-particle quantum walks applied to the graph isomorphism problem](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.81.052313)

---

# References - 2

- Image sources:
  - [Title Slide Variations 1 & 2](#title-slide-variation-1)
    - [pxfuel](https://www.pxfuel.com/en/desktop-wallpaper-iqgfc)
    - [HiClipart](https://www.hiclipart.com/free-transparent-background-png-clipart-pgsow)
  - [Image Slide 1](#image-slide-1) - [ShutterStock](https://www.shutterstock.com/image-vector/set-funny-eggs-on-brown-background-2118472637)
  - [Image Slide 2](#image-slide-2) - [Earth Stock photos by Vecteezy](https://www.vecteezy.com/photo/25436091-paper-art-world-environment-and-earth-day-concept-green-planet-environmental-protection-save-the-planet-generate-ai)

---

<!-- _paginate: false -->
<!-- _footer: '' -->
<style scoped>
  h1 {
    color: #18716F;
    margin-top: 250px;
    text-align: center;
  }
</style>

# _Questions?_

---

<!-- _paginate: false -->
<!-- _footer: '' -->
<style scoped>
  h1 {
    color: #18716F;
    /* margin-top: 250px; */
    /* text-align: center; */
  }
</style>

<!-- NOTE: This is for after the talk, to add some extra details based on feedback if needed -->

# Post-talk Questions

- Question 1
  - Some details
- Question 2
  - Some details

---

<!-- NOTE: LEAVE THIS SLIDE AS-IS -->
<!-- _class: lead -->
<!-- _footer: '' -->
<!-- _paginate: false -->
<!-- _backgroundColor: #000 -->
<!-- _backgroundImage: url('./media/slide_bg_light.png') -->

![bg right:55%](./media/qrcode_smlab.svg)

<!-- NOTE: Multiple images added here - one in BG, one in FG -->
<style scoped>
  h2 {
    font-size: 50px;
    color: #f35422;
  }
  .socials {
    margin-top: 30px;
    width: 100%;
    text-align: center;
    vertical-align: middle;
    color: red;
  }
  .socials a {
    text-align: center;
  }
  .socials a img {
    width: 60px;
    height: 60px;
  }
  a {
    margin: 15px;
    font-size: 40px;
    color: #fff;
    text-decoration: none;
  }
</style>

<h2> Subhankar Mishra Lab Weekly Talks </h2>

<!-- Socials -->
<span class="socials">
<!-- Lab-talks website -->
  <a href="https://smlab.niser.ac.in/labtalks/" target="_blank">
    <img src="./media/web.png" alt="Website">
  </a>
  <!-- Lab-talks GitHub -->
  <a href="https://github.com/JeS24/smlab-talks/" target="_blank">
    <img src="./media/gh.svg" alt="GitHub">
  </a>
  <!-- Lab Twitter -->
  <a href="https://twitter.com/mishra_lab" target="_blank">
    <img src="./media/twitter.svg" alt="Twitter">
  </a>
  <!-- Lab public email -->
  <a href="mailto:smlab@niser.ac.in">
    <img src="./media/email.png" alt="Mail">
  </a>
</span>
