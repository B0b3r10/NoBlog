from django.db import models


class Post(models.Model):
    title = models.CharField(primary_key=True, max_length=100)
    # текст поста
    body = models.TextField()

    def __str__(self):
        return self.title
