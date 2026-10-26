package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class EmojiModel {
    private Emoji[] emoji_list;

    public class Emoji {
        private String name;
        private int unicode;

        public Emoji() {
        }

        public String getName() {
            return this.name;
        }

        public int getUnicode() {
            return this.unicode;
        }

        public void setName(String str) {
            this.name = str;
        }

        public void setUnicode(int i) {
            this.unicode = i;
        }

        public String toString() {
            return "EmojiModel{name='" + this.name + "', unicode=" + this.unicode + '}';
        }
    }

    public Emoji[] getEmoji_list() {
        return this.emoji_list;
    }

    public void setEmoji_list(Emoji[] emojiArr) {
        this.emoji_list = emojiArr;
    }
}
