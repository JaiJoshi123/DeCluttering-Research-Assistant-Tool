from django.views.generic import TemplateView
from django.shortcuts import render, redirect
from django.contrib.auth import get_user_model
import random
import pandas as pd
import numpy as np
from . import funcs
from datetime import datetime

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client.browser_buddy


USER_IDS = [
    '2b6c0f514c2f2b04ad3c4583407dccd0810469ee', '77959baaa9895a7e2bdc9297f8b27c1b6f2cb52a', '2f5c7feae533ce046f2cb16fb3a29fe00528ed66', 'a37adec71b667b297ed2440a9ff7dad427c7ac85', '8510a5010a5d4c89f5b07baac6de80cd12cfaf93', 'f8c978bcf2ae2fb8885814a9b85ffef2f54c3c76', '284d0c17905de71e209b376e3309c0b08134f7e2', '18e7255ee311d4bd78f5993a9f09538e459e3fcc', 'd9032ff68d0fd45dfd18c0c5f7324619bb55362c', '276d9d8ca0bf52c780b5a3fc554fa69e74f934a3', 'c60bb0a50c324dad0bffd8809d121246baef372b', '56832a697cb6dbce14700fca18cffcced367057f', 'b2d2c70ed5de62cf8a1d4ded7dd141cfbbdd0388', 'ceef2a24a2a82031246814b73e029edba51e8ea9', '8dc8d7ec2356b1b106eb3d723f3c234e03ab3f1e', 'e38f123afecb40272ba4c47cb25c96a9533006fa', '53db7ac77dbb80d6f5c32ed5d19c1a8720078814', '6c14453c049b1ef4737b08d56c480419794f91c2', 'fd824fc62b4753107e3db7704cd9e8a4a1c961f1', 'c45f9495a76bf95d2633444817f1be8205ad542d', '12bb8a9740400ced27ae5a7d4c990ac3b7e3c77d', '3427a5a4065625363e28ac8e85a57a9436010e9c', '0d644205ecefdef33e3346bb3551f5e68dc57c58', '497935037e41a94d2ae02488d098c7abda9a30bc', '015aaf617598e413a35d6d2249e26b7f3c40adb7', 'e90de4b883d9de64a47774ad7ad49ca6fd69d4fe', 'db1c400ffb74f14390deba2140bd31d2e1dc5c4e', '7dc02db8b76fffbdfe29542da672d4d5fd5ed4ae', '2e205a44014ca7bdbf07fc32f3c9d17699671d03', 'b2926913d95598ec0c007746d693fe3e466ff2d4', '4070b8d82484ed99cdb9bbc2ebf4e9aca06fd934', '463878695aac3acc71e9d7c18e7a3b5d8e1a5456', 'f1ccb4d9d8446f26c6c8ee2a135782f984526860', '6f2a2814638cb70081ef84e149619eb3f4490f4f', '7b1389c3204f4205132973e68dfe2d20912df0f2', '11304ae794b552e6c929654daaea245e5b57f03b', '19c7d87e50dd9da96c7d2a980139df1497b94247', '665470e2d4eb76437965ec71e52b41d55f15a08d', 'c9086fbe74843c4792d030260be1499c558edc03', '157d0aba8d75f1c72e5428e4a64a51906008a43a', '66a82c0d9e01c515d058d9bec352a4d8e6db0867', '83872a57e34cb69fe4ee76908b004440d410ed22', '02e21c6cf9a4ac6772ef58fc6bbd86c3a428d6a4', '6d67ed7e58b2992d875f393506e0ee28c6993366', '3bbca9c41c66c4ecc46bc14eac7b7ab1ab0a5bea', 'b96a4f2e92d8572034b1e9b28f9ac673765cd074', '40b1ef41687b2681d7d92221ce074291d94289f0', 'fb882a39e938a9c7b7d6f13c8dba0e03898fbe31', '2daffd59992d5e8f286c3c2344a206eb477ce6fe', '7f08a630ce7028ca92a7ad97c71e7091de2baaa1', 'c8afd6f4620184042cc48ca0eba9a657ac89e90e', '8a19c743688b2c36c8acd9931fe3974a598cb3d5', '85bd9a5b292b2f0dbcbb54ca4fc03ee15b668ac3', '1588af175b283915f597fc4719cbb2c8621c4fc2', '877584ade75e135b9943d5214370def9b4063bd8', 'a7de4b804d75097dadbad34a9cb4aab060eea248', '0e321f3b0387d0e8a5b46ff55ff3b4f820fa38b9', '363cb98a087e4a3eb6890fd1af2d418116f85ff8', 'd385b6c44711688ce2a33eb596495bf9243e1750', 'be4dd9da5399f864c176adf91134ba17a5ff1746', '3b57bde8f07468e3d912559126b4e8bf5984e992', '27084f1d0a1be41e3a6402cd7a4ac5e73171d174', '5762369769eacb15c10da38989a62a3f229ae029', '12815feeacc6f27dff5b3441a54418d2d51001ef', '4e471e4263ac5d11b9ad2c06bc673d76b594a02a', 'e4bfcfa12894e6b1b51e48b9aa117b9402847f0c', 'd5843ed71361c87b364f578f20a48101289d60f9', '0e6cfaae829cd1ff109f01e7487b83be53cfc2f3', '6cce7568da5452718e1a3702edffac34a8da74ec', 'c1ce968fdc09933181e21105e9fa0bdf7d6bbcb6', '3f8cc93d9ab98284d874e99c39aca0e4fc497792', '67189216e738ace749c97ffc9c92fda82f5117a0', 'bb7589e21750a3ee9e82b20d6f4779aa25849620', 'e654f8ce61fbb1e536a5eee4e490cdcb97f68077', 'be57ea88bf277c74945357e89cd795a471b6df69', 'e29e7c231a152bef11d0121d2ad4b0696f535423', '20cd78fb99bf664deac4548c7f6c321d47d029a0', 'b2ad1860281633817f584ac7d6cee322ad96bfbd', '683a1cf44d2a711ec83a5b57890474f32ada3bee', '146d98d063d39711c988c9065e0e30374a5b5798', '0c8ecfc9fa920e5495e1b603a8de5b1238262146', '564dd834427fb4a9410f670a368bc73978e9653d', '2f7f6b811d7e18d143594917993e2b0c72a535e2', '108e0fbb23af760d1fc120a2171ea21b203794c7', 'a5e5a52ab275f02a72d5c2fa61687e73d895855a', '38d63557c2ba70ceded69d0666dce709e82b7038', 'dc716911a33e23c23667a082667f40a4a6e4671d', '55f23ecebba7404df1f77d86bd7235f067f7e6c5', '11a3c57739fda7439f5a6419055c7990a510786e', 'a0e44bd2c07b772255812e1a3ad29ccc09886597', 'b000e61505873337d072aabf4be71ff789de4b68', '6fc7727ccb3cf81a9460c5625d73a87b9da5e7bf', '2ff319cba3204a2cc83349ec78aa478099ec32d6', '40cd8c911f4ffb8c1798441aef46ecc3831a43d1', '80b0cde82a3ada16d053b542243ba80b03ce29ae', '5fde1823628b555eecd3885ca01b84c2f91e4e58', '50b34cc22db66d2e88c39bdc4e666c91ccbb0a65', 'aedcfe1a2d62760adc7db57a1489deb2e78f1858', 'b05467cb9a2b2dd4b9f7ce7a1c502d432de9775c', '21e10cfb7b3208992a1fdf02c82bb4cdb849806c']

UserModel = get_user_model()

def index(request):
    if(request.user.is_anonymous == False):
        user = UserModel.objects.get(email=request.user.email)
        if(user.is_new):
            all_users = UserModel.objects.values()
            l = []
            for u in all_users:
                if(u["email"]!= "browserbuddy@gmail.com"):
                    l.append(u["user_id_no"])
            i = random.randint(0,100)
            while(USER_IDS[i] in l):
                i = random.randint(0,100)
            user.user_id_no = USER_IDS[i]
            user.is_new = False
            user.save()
            request.user = user
        user_article_ids = db.user_items_interaction.find({"email": user.user_id_no})
        articles = []
        for article in user_article_ids:
            content = db.articles.find_one({"article_id": str(article.get("article_id")).replace(".0","")})
            if(content):
                articles.append({
                    "id": article.get("_id"),
                    "article_id": article.get("article_id"),
                    "title": article.get("title"),
                    "description": content["doc_description"],
                    "status": content["doc_status"],
                    "body": content["doc_body"]
                })
        additional_articles_count = db.user_articles.count_documents({"user": request.user.email})

        user_df = funcs.createdf()
        u = user_df.loc[user_df['email'] == request.user.user_id_no]
        user_id = u.iloc[0]["user_id"]
        sim_users = funcs.find_similar_users(user_id, user_df)
        print(sim_users[:10])

        similar_users = []
        for usr_id in sim_users[:10]:
            ux = user_df.loc[user_df['user_id'] == usr_id]
            similar_users.append({
                "email": ux.iloc[0]["email"],
                "user_id": usr_id,
            })
        
        return render(request, "web/index_1.html", {
            "article_count": len(articles) + additional_articles_count,
            "articles": articles,
            "similar_users": similar_users,
        })
    else:
        return redirect("account_login")


def recommendations(request):
    if(request.user.is_anonymous == False):
        user_df = funcs.createdf()
        u = user_df.loc[user_df['email'] == request.user.user_id_no]
        user_id = u.iloc[0]["user_id"]
        print(user_id)
        rec_names_id = funcs.user_user_recs(user_id,user_df,20)
        print(rec_names_id)
        print(rec_names_id[0].replace(".0",""))
        rec_names = []
        for r_id in rec_names_id:
            r = db.articles.find_one({"article_id": str(r_id.replace(".0",""))})
            if(r):
                rec_names.append({
                    "title": r.get("doc_full_name"),
                    "description": r.get("doc_description"),
                    "status": r.get("doc_status"),
                    "body": r.get("doc_body")
                })
            
        return render(request, "web/recommendations.html", {
                "recommendations": rec_names,
                "buckets": ["Deep Learning", "Artificial Intelligence", "Data analytics", "Machine Learning","IoT", "Computing", "Postgres"]
        })
    else:
        return redirect("account_login")

def previous_articles(request):
    if(request.user.is_anonymous == False):
        user_article_ids = db.user_items_interaction.find({"email": request.user.user_id_no})
        articles = []
        for article in user_article_ids:
            content = db.articles.find_one({"article_id": str(article.get("article_id")).replace(".0","")})
            if(content):
                articles.append({
                    "id": article.get("_id"),
                    "article_id": article.get("article_id"),
                    "title": article.get("title"),
                    "description": content["doc_description"],
                    "status": content["doc_status"],
                    "body": content["doc_body"]
                })
        additional_articles = db.user_articles.find({"user": request.user.email})
        for article in additional_articles:
            articles.append({
                    "title": article.get("title"),
                    "description": article.get("summary"),
                    "URL": article.get("URL"),
                })
        return render(request, "web/previous_articles.html", {
                "article_count": len(articles),
                "articles": articles,
            })
    else:
        return redirect("account_login")

def article_summarizer(request):
    if(request.user.is_anonymous == False):
        if(request.method == "POST"):
            print('POST')
            title,summary = funcs.get_summary(request.POST["URL"])
            return render(request, "web/article_summarizer.html", {
                "summary": summary,
                "title": title,
                "URL": request.POST["URL"],
            })
        else:
            return render(request, "web/article_summarizer.html", {})
    else:
        return redirect("account_login")

def add_article(request):
    if(request.method=="POST"):
        res = db.user_articles.insert_one(
            {
                'summary': request.POST["summary"],
                'user': request.user.email,
                'title': request.POST["title"], 
                'dateAdded': datetime.now(),
                'URL': request.POST.get('URL',False)
            }
        )
        print(res)
        return redirect("previous_articles")

def article_summarizer_2(request):
    if(request.user.is_anonymous == False):
        if(request.method == "POST"):
            print('POST')
            summary = funcs.get_summary_2(request.POST.get("content",""))
            return render(request, "web/article_summarizer.html", {
                "summary": summary,
                "title": request.POST["title"],
             
            })
        else:
            return render(request, "web/article_summarizer.html", {})
    else:
        return redirect("account_login")