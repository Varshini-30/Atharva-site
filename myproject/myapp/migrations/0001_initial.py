# Generated by Django 5.0.6 on 2024-06-23 12:49

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Feedback",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=200)),
                ("phone", models.CharField(max_length=15)),
                ("email", models.EmailField(max_length=254)),
                ("location", models.CharField(max_length=400)),
                ("detail", models.CharField(max_length=800)),
            ],
            options={
                "verbose_name": "Feedback List",
                "db_table": "Feedback",
            },
        ),
    ]
