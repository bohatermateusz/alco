package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class ChatUpModel extends HttpModel {
    private FirstChatModel[] firstChat;
    private int page;
    private int pageSize;
    private int totalCount;
    private int totalPage;

    public FirstChatModel[] getFirstChat() {
        return this.firstChat;
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

    public void setFirstChat(FirstChatModel[] firstChatModelArr) {
        this.firstChat = firstChatModelArr;
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
        return "ChatUpModel{firstChat=" + Arrays.toString(this.firstChat) + ", page=" + this.page + ", pageSize=" + this.pageSize + ", totalCount=" + this.totalCount + ", totalPage=" + this.totalPage + '}';
    }
}
