package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class TopicCategoriesModel extends HttpModel {
    private TopicType[] discover;
    private TopicType[] menu;
    private TopicType[] post;
    private TAGType[] tag;

    public class TAGType {
        private String code;
        private String title;

        public TAGType() {
        }

        public String getCode() {
            return this.code;
        }

        public String getTitle() {
            return this.title;
        }

        public void setCode(String str) {
            this.code = str;
        }

        public void setTitle(String str) {
            this.title = str;
        }

        public String toString() {
            return "TAGType{code='" + this.code + "', title='" + this.title + "'}";
        }
    }

    public class TopicType {
        private String icon;
        private String iconUrl;
        private String id;
        private String title;

        public TopicType() {
        }

        public String getIcon() {
            return this.icon;
        }

        public String getIconUrl() {
            return this.iconUrl;
        }

        public String getId() {
            return this.id;
        }

        public String getTitle() {
            return this.title;
        }

        public void setIcon(String str) {
            this.icon = str;
        }

        public void setIconUrl(String str) {
            this.iconUrl = str;
        }

        public void setId(String str) {
            this.id = str;
        }

        public void setTitle(String str) {
            this.title = str;
        }

        public String toString() {
            return "TopicType{id='" + this.id + "', title='" + this.title + "', icon='" + this.icon + "', iconUrl='" + this.iconUrl + "'}";
        }
    }

    public TopicType[] getDiscover() {
        return this.discover;
    }

    public TopicType[] getMenu() {
        return this.menu;
    }

    public TopicType[] getPost() {
        return this.post;
    }

    public TAGType[] getTag() {
        return this.tag;
    }

    public void setDiscover(TopicType[] topicTypeArr) {
        this.discover = topicTypeArr;
    }

    public void setMenu(TopicType[] topicTypeArr) {
        this.menu = topicTypeArr;
    }

    public void setPost(TopicType[] topicTypeArr) {
        this.post = topicTypeArr;
    }

    public void setTag(TAGType[] tAGTypeArr) {
        this.tag = tAGTypeArr;
    }

    public String toString() {
        return "TopicCategoriesModel{menu=" + Arrays.toString(this.menu) + ", post=" + Arrays.toString(this.post) + ", discover=" + Arrays.toString(this.discover) + ", tag=" + Arrays.toString(this.tag) + '}';
    }
}
