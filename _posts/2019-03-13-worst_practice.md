---
title: The worst practice for software engineering
categories: [Technology]
---

People will suffer a lot from bad design or bad code in software engineering. Today I'd like to share a number of the worst practices.

<!-- more -->

### Put code in MS Word, or PDF file

> Hey, the code in Word is also colorful. By the way, I can cancel the modification easily. Oh, even cooperating with friends. That's interesting.

Oh, come on. Don't tell me that your code is for exhibiting?

All right. If your teachers tell you to do so, that means they may not run your code. Of course, they won't do the duplicate checking, either. You may need to find more resource on the Internet for promotion. Don't count on school.

### Code is only saved in local and live server without version control

> You know what, 'git' means !@#$%^&*

Congratulations if your company still run well. And hope the person who takes over your work won't curse you for a month. (Oh, he may not know your contract since you left nothing.)

### data1, data2, path1, path2, name1, list0, fun1...

> Do you know how many time the programmers spend on naming?

Then, you really need a good editor. Although CPU doesn't care the names, I think you are working with human beings.

### Write duplicate code everywhere

> Hey, guess how many lines of code I have write in one day?

Your boss won't pay you more money due to the lines of code you write. But you really waste a lot of time and you will waste more time. Since every time you change one place, you need to change many places and you may forget where to change.

Do you know there is something we call "function"?

### What is loop?

> Type `staff.append('xxx')` for 100 times in one minute.

Fine, I know your typing is very fast. But do you know sometimes we use the loop?

### Print everything to std::out

> That's very clear. You see, everything goes well.

Once your code raises an exception, you can only ask the God for help. For every project, logging will give you hints when the code goes wrong. Besides, that's not difficult to replace print with logging. There are also lots of tools to analyse logs.

### Save everything to local files

> Now I have everything stored.

Later you may lost everything. Remember to use database. Lots of people develop it for this purpose.

### Push everything to code repository

> Now, everything is covered by version control!

I'll pull down your code and deploy it. Wait, it's 2.33 GB? Are you kidding me? Why there are so many binray files and logs?

I know there is large file system for data store. But apparently, these files should be ignored.

### Leave README file blank

> Actually, I don't have README file :) Very mysterious, right?

LOL. I don't even show you my code. Am I mysterious?

### Leave useless code everywhere

> These code may be useful next time.

You really need to master version control.

### No documents, no comments, no tests

> I'll add them when the project is finished.

Really? :)

### Use HashMap everywhere

> I just learned that it's O(1)! `map[0] = 'a'; map[1] = 'b'; map[2] = 'c';`

Clever. But you only need an array.

## Useful resources

- [A Guide to Naming Variables](https://a-nickels-worth.blogspot.com/2016/04/a-guide-to-naming-variables.html)
- [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)
- [Git cheat sheet](https://education.github.com/git-cheat-sheet-education.pdf)

