package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class TopicListModel extends HttpModel {
    private Integer page;
    private Integer pageSize;
    private String requestId;
    private TopicModel[] topics;
    private Integer totalCount;
    private Integer totalPage;

    public Integer getPage() {
        return this.page;
    }

    public Integer getPageSize() {
        return this.pageSize;
    }

    public String getRequestId() {
        return this.requestId;
    }

    public TopicModel[] getTopics() {
        return this.topics;
    }

    public Integer getTotalCount() {
        return this.totalCount;
    }

    public Integer getTotalPage() {
        return this.totalPage;
    }

    public void setPage(Integer num) {
        this.page = num;
    }

    public void setPageSize(Integer num) {
        this.pageSize = num;
    }

    public void setRequestId(String str) {
        this.requestId = str;
    }

    public void setTopics(TopicModel[] topicModelArr) {
        this.topics = topicModelArr;
    }

    public void setTotalCount(Integer num) {
        this.totalCount = num;
    }

    public void setTotalPage(Integer num) {
        this.totalPage = num;
    }
}
