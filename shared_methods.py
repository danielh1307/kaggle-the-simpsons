from PIL import Image

all_labels = ["abraham_grampa_simpson",
              "agnes_skinner",
              "apu_nahasapeemapetilon",
              "barney_gumble",
              "bart_simpson",
              "carl_carlson",
              "charles_montgomery_burns",
              "chief_wiggum",
              "cletus_spuckler",
              "comic_book_guy",
              "disco_stu",
              "edna_krabappel",
              "fat_tony",
              "gil",
              "groundskeeper_willie",
              "homer_simpson",
              "kent_brockman",
              "krusty_the_clown",
              "lenny_leonard",
              "lionel_hutz",
              "lisa_simpson",
              "maggie_simpson",
              "marge_simpson",
              "martin_prince",
              "mayor_quimby",
              "milhouse_van_houten",
              "miss_hoover",
              "moe_szyslak",
              "ned_flanders"
              ]


def show_image_by_path(_image_path: str) -> None:
    image = Image.open(_image_path)
    image.show()
