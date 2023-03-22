from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('recommendations', views.recommendations ,name="recommendations"),
    path('previous_articles', views.previous_articles,name="previous_articles"),
    path('article_summarizer', views.article_summarizer,name="article_summarizer"),
    path('article_summarizer_2', views.article_summarizer_2 ,name="article_summarizer_2"),
    path('add_article', views.add_article,name="add_article"),
]