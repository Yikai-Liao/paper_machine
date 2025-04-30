## Subhankar Mishra Lab Weekly Talks

This repository stores the files of the talks & presentations by our lab members and interns on various Machine Learning or Computer Science topics.

The same is hosted at https://smlab.niser.ac.in/labtalks/.

> [!TIP]
> See [`./smlab-talk-templates`](./smlab-talk-templates) for the Marp templates for the talks. Contact the lab members for acquiring access to the fonts.

> [!NOTE]
> 1. For people looking to edit the site, please refer to [`AstroPaper.README.md`](./AstroPaper.README.md) for the setup and configuration of this website.
> 2. For the Google AppScript that controls the automated mails, please refer to [`auto-mails-appscript.gs`](./AppScript/auto-mails-appscript.gs).


### 📚 How to add new talks?

1. Fork this repository.
2. Clone the forked repository.
3. `cd` into the repository.
4. Copy the `./src/content/blog/YYYY-MM-DD--FirstName--ShortTalkTitle.md.template` file to a new file with the appropriate filename and `.md` extension.
5. Fill in the details in the new file. Follow the `TODO:` tags in the template.
6. Delete the `TODO:` tags once you have filled in the details.
7. For the frontmatter, ensure that there is no `:` in the title, author, or other tags. If there are any, replace them with ` - `.
8. Add your slides to the `./public/assets/slides/` folder. The slides should be named `YYYY-MM-DD--FirstName--ShortTalkTitle.pdf` (same as the post filename with a `.pdf` extension).
9. Push the changes to your forked repository.
10. Create a pull request to the main repository.
11. Once the pull request is merged, the website will be updated automatically. 🥳

### ✨ Feedback & Suggestions

For queries, suggestions, feedback, or issues, please [create an issue](https://github.com/JeS24/smlab-talks/issues/new).

### 📜 License

Content is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), Copyright © 2024 Subhankar Mishra Lab, National Institute of Science Education and Research, Bhubaneswar, India

Template licensed under the MIT License, Copyright © 2024 [Sat Naing](https://satnaing.dev)

---
Thanks [Sat Naing](https://satnaing.dev) 👨🏻‍💻 and [contributors](https://github.com/satnaing/astro-paper/graphs/contributors) for the template. This repo adds some extra features, such as support for a `base` URL, fuzzy search extended to authors, and more.
