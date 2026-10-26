package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class InteractiveModel extends HttpModel {
    private Interactive[] interactive;
    private int page;
    private int pageSize;
    private int totalCount;
    private int totalPage;

    public class Interactive {
        private String age;
        private String constellation;
        private String date;
        private Integer followed;
        private String gender;
        private String gradeIcon;
        private String headIcon;
        private int id;
        private int king;
        private String name;
        private String resultContent;
        private String resultId;
        private String resultImage;
        private String sourceContent;
        private int sourceDelete;
        private String sourceId;
        private String sourceImage;
        private int status;
        private String type;
        private String typeNote;
        private int userId;

        public Interactive() {
        }

        public String getAge() {
            return this.age;
        }

        public String getConstellation() {
            return this.constellation;
        }

        public String getDate() {
            return this.date;
        }

        public Integer getFollowed() {
            return this.followed;
        }

        public String getGender() {
            return this.gender;
        }

        public String getGradeIcon() {
            return this.gradeIcon;
        }

        public String getHeadIcon() {
            return this.headIcon;
        }

        public int getId() {
            return this.id;
        }

        public int getKing() {
            return this.king;
        }

        public String getName() {
            return this.name;
        }

        public String getResultContent() {
            return this.resultContent;
        }

        public String getResultId() {
            return this.resultId;
        }

        public String getResultImage() {
            return this.resultImage;
        }

        public String getSourceContent() {
            return this.sourceContent;
        }

        public int getSourceDelete() {
            return this.sourceDelete;
        }

        public String getSourceId() {
            return this.sourceId;
        }

        public String getSourceImage() {
            return this.sourceImage;
        }

        public int getStatus() {
            return this.status;
        }

        public String getType() {
            return this.type;
        }

        public String getTypeNote() {
            return this.typeNote;
        }

        public int getUserId() {
            return this.userId;
        }

        public void setAge(String str) {
            this.age = str;
        }

        public void setConstellation(String str) {
            this.constellation = str;
        }

        public void setDate(String str) {
            this.date = str;
        }

        public void setFollowed(Integer num) {
            this.followed = num;
        }

        public void setGender(String str) {
            this.gender = str;
        }

        public void setGradeIcon(String str) {
            this.gradeIcon = str;
        }

        public void setHeadIcon(String str) {
            this.headIcon = str;
        }

        public void setId(int i) {
            this.id = i;
        }

        public void setKing(int i) {
            this.king = i;
        }

        public void setName(String str) {
            this.name = str;
        }

        public void setResultContent(String str) {
            this.resultContent = str;
        }

        public void setResultId(String str) {
            this.resultId = str;
        }

        public void setResultImage(String str) {
            this.resultImage = str;
        }

        public void setSourceContent(String str) {
            this.sourceContent = str;
        }

        public void setSourceDelete(int i) {
            this.sourceDelete = i;
        }

        public void setSourceId(String str) {
            this.sourceId = str;
        }

        public void setSourceImage(String str) {
            this.sourceImage = str;
        }

        public void setStatus(int i) {
            this.status = i;
        }

        public void setType(String str) {
            this.type = str;
        }

        public void setTypeNote(String str) {
            this.typeNote = str;
        }

        public void setUserId(int i) {
            this.userId = i;
        }

        public String toString() {
            return "Interactive{id=" + this.id + ", userId=" + this.userId + ", king=" + this.king + ", gradeIcon='" + this.gradeIcon + "', name='" + this.name + "', gender='" + this.gender + "', age='" + this.age + "', constellation='" + this.constellation + "', headIcon='" + this.headIcon + "', followed=" + this.followed + ", type='" + this.type + "', typeNote='" + this.typeNote + "', sourceId='" + this.sourceId + "', sourceContent='" + this.sourceContent + "', sourceImage='" + this.sourceImage + "', sourceDelete=" + this.sourceDelete + ", resultId='" + this.resultId + "', resultContent='" + this.resultContent + "', resultImage='" + this.resultImage + "', date='" + this.date + "', status=" + this.status + '}';
        }
    }

    public Interactive[] getInteractive() {
        return this.interactive;
    }

    public int getPage() {
        return this.page;
    }

    public int getPageSize() {
        return this.pageSize;
    }

    public int getTotalCount() {
        return this.totalCount;
    }

    public int getTotalPage() {
        return this.totalPage;
    }

    public void setInteractive(Interactive[] interactiveArr) {
        this.interactive = interactiveArr;
    }

    public void setPage(int i) {
        this.page = i;
    }

    public void setPageSize(int i) {
        this.pageSize = i;
    }

    public void setTotalCount(int i) {
        this.totalCount = i;
    }

    public void setTotalPage(int i) {
        this.totalPage = i;
    }

    public String toString() {
        return "InteractiveModel{interactive=" + Arrays.toString(this.interactive) + ", page=" + this.page + ", pageSize=" + this.pageSize + ", totalCount=" + this.totalCount + ", totalPage=" + this.totalPage + '}';
    }
}
