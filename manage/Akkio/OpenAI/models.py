from django.db import models


# Create your models here.

#For sending the mail
# from django.core.mail import send_mail
# from django.conf import settings

# def send_welcome_email(to_email, subject, message):
#     from_email = settings.EMAIL_HOST_USER
#     send_mail(subject, message, from_email, [to_email])
    


from django.db import models

class GeneratedImage(models.Model):
    email = models.EmailField()
    selected_style = models.CharField(max_length=100)
    selected_room_color = models.CharField(max_length=100)
    selected_room_type = models.CharField(max_length=100)
    #number_of_room_designs = models.IntegerField()
    image = models.ImageField(upload_to='generated_images/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Image for {self.email} - {self.created_at}'






