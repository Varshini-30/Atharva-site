from django.db import models

# Create your models here.

class Enquiry(models.Model):
    name = models.CharField(max_length=200)
    phone = models.CharField(max_length=15)
    email = models.EmailField()
    location = models.CharField(max_length=400)
    detail = models.CharField(max_length=800)

    def __str__(self):
        return self.name
    
    class Meta: 
        db_table = 'Enquiry'
        verbose_name = 'Enquiry List'