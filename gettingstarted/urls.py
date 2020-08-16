from django.urls import path, include

from django.contrib import admin

admin.autodiscover()

import hello.views

# To add a new path, first import the app:
# import blog
#
# Then add the new path:
# path('blog/', blog.urls, name="blog")
#
# Learn more here: https://docs.djangoproject.com/en/2.1/topics/http/urls/

urlpatterns = [
    path("", hello.views.index, name="index"),
    path("results/", hello.views.results, name="results"),
    path("details/1", hello.views.details, name="details"),
    path("sharestory/", hello.views.sharestory, name="sharestory"),
    path("admin/", admin.site.urls),
]
