from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    # add additional fields in here
    is_new = models.BooleanField(default=True)
    user_id_no = models.CharField(max_length=254, null=True, blank=True)
    def __str__(self):
        return self.email
