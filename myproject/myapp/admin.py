from django.contrib import admin
from .models import Enquiry
# Register your models here.

class Enquiryadmin(admin.ModelAdmin):
    list_display=('id','name', 'phone', 'email', 'location', 'detail')

    # def has_add_permission(self, request): #to remove add Enquiry button
    #     return False

admin.site.register(Enquiry,Enquiryadmin)