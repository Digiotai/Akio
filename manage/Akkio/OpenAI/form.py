from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django import forms

class PlaceholderMixin:
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
            field_names = [field_name for field_name, _ in self.fields.items()]
            for field_name in field_names:
                field = self.fields.get(field_name)
                field.widget.attrs.update({'placeholder': field.label})
        except Exception as e:
            print(e)

class CreateUserForm(PlaceholderMixin, UserCreationForm):
    class Meta:
        try:
            model = User
            fields =['username','email','password1','password2']
        except Exception as e:
            print(e)
