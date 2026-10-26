package com.hismith.hismithapp.model;

import android.text.TextUtils;
import com.hismith.hismithapp.http.HttpModel;
import java.io.Serializable;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class MessageModel extends HttpModel implements Serializable {
    private int firstChatCount;
    private int interactiveCount;
    private Message[] message;
    private int noticeCount;
    private RevokeIds[] rejectIds;
    private RevokeIds[] revokeIds;

    public class Message implements Serializable {
        private String age;
        private String alias;
        private int androidRead;
        private int androidReadCount;
        private int androidSend;
        private String constellation;
        private String content;
        private String date;
        private int gender;
        private String gradeIcon;
        private String headIcon;
        private int king;
        private String localPath;
        private int msgId;
        private String name;
        private int receiverId;
        private int referenceId;
        private ReferenceMsg referenceMsg;
        private int reject;
        private int revoke;
        private int status;
        private String type;
        private int userId;
        private int voiceLength;

        public Message() {
        }

        public String getAge() {
            return this.age;
        }

        public String getAlias() {
            return !TextUtils.isEmpty(this.alias) ? this.alias : this.name;
        }

        public int getAndroidRead() {
            return this.androidRead;
        }

        public int getAndroidReadCount() {
            return this.androidReadCount;
        }

        public int getAndroidSend() {
            return this.androidSend;
        }

        public String getConstellation() {
            return this.constellation;
        }

        public String getContent() {
            return this.content;
        }

        public String getDate() {
            return this.date;
        }

        public int getGender() {
            return this.gender;
        }

        public String getGradeIcon() {
            return this.gradeIcon;
        }

        public String getHeadIcon() {
            return this.headIcon;
        }

        public int getKing() {
            return this.king;
        }

        public String getLocalPath() {
            return this.localPath;
        }

        public int getMsgId() {
            return this.msgId;
        }

        public String getName() {
            return this.name;
        }

        public int getReceiverId() {
            return this.receiverId;
        }

        public int getReferenceId() {
            return this.referenceId;
        }

        public ReferenceMsg getReferenceMsg() {
            return this.referenceMsg;
        }

        public int getReject() {
            return this.reject;
        }

        public int getRevoke() {
            return this.revoke;
        }

        public int getStatus() {
            return this.status;
        }

        public String getType() {
            return this.type;
        }

        public int getUserId() {
            return this.userId;
        }

        public int getVoiceLength() {
            return this.voiceLength;
        }

        public void setAge(String str) {
            this.age = str;
        }

        public void setAlias(String str) {
            this.alias = str;
        }

        public void setAndroidRead(int i) {
            this.androidRead = i;
        }

        public void setAndroidReadCount(int i) {
            this.androidReadCount = i;
        }

        public void setAndroidSend(int i) {
            this.androidSend = i;
        }

        public void setConstellation(String str) {
            this.constellation = str;
        }

        public void setContent(String str) {
            this.content = str;
        }

        public void setDate(String str) {
            this.date = str;
        }

        public void setGender(int i) {
            this.gender = i;
        }

        public void setGradeIcon(String str) {
            this.gradeIcon = str;
        }

        public void setHeadIcon(String str) {
            this.headIcon = str;
        }

        public void setKing(int i) {
            this.king = i;
        }

        public void setLocalPath(String str) {
            this.localPath = str;
        }

        public void setMsgId(int i) {
            this.msgId = i;
        }

        public void setName(String str) {
            this.name = str;
        }

        public void setReceiverId(int i) {
            this.receiverId = i;
        }

        public void setReferenceId(int i) {
            this.referenceId = i;
        }

        public void setReferenceMsg(ReferenceMsg referenceMsg) {
            this.referenceMsg = referenceMsg;
        }

        public void setReject(int i) {
            this.reject = i;
        }

        public void setRevoke(int i) {
            this.revoke = i;
        }

        public void setStatus(int i) {
            this.status = i;
        }

        public void setType(String str) {
            this.type = str;
        }

        public void setUserId(int i) {
            this.userId = i;
        }

        public void setVoiceLength(int i) {
            this.voiceLength = i;
        }

        public String toString() {
            return "Message{msgId=" + this.msgId + ", userId=" + this.userId + ", receiverId=" + this.receiverId + ", king=" + this.king + ", gradeIcon='" + this.gradeIcon + "', name='" + this.name + "', gender=" + this.gender + ", age='" + this.age + "', constellation='" + this.constellation + "', headIcon='" + this.headIcon + "', type='" + this.type + "', voiceLength=" + this.voiceLength + ", content='" + this.content + "', date='" + this.date + "', revoke=" + this.revoke + ", reject=" + this.reject + ", androidRead=" + this.androidRead + ", androidReadCount=" + this.androidReadCount + ", androidSend=" + this.androidSend + ", status=" + this.status + ", localPath='" + this.localPath + "', alias='" + this.alias + "', referenceId=" + this.referenceId + ", referenceMsg=" + this.referenceMsg + '}';
        }
    }

    public class RevokeIds {
        private String msgId;
        private String type;
        private String userId;

        public RevokeIds() {
        }

        public String getMsgId() {
            return this.msgId;
        }

        public String getType() {
            return this.type;
        }

        public String getUserId() {
            return this.userId;
        }

        public void setMsgId(String str) {
            this.msgId = str;
        }

        public void setType(String str) {
            this.type = str;
        }

        public void setUserId(String str) {
            this.userId = str;
        }

        public String toString() {
            return "RevokeIds{msgId='" + this.msgId + "', userId='" + this.userId + "', type='" + this.type + "'}";
        }
    }

    public int getFirstChatCount() {
        return this.firstChatCount;
    }

    public int getInteractiveCount() {
        return this.interactiveCount;
    }

    public Message[] getMessage() {
        return this.message;
    }

    public int getNoticeCount() {
        return this.noticeCount;
    }

    public RevokeIds[] getRejectIds() {
        return this.rejectIds;
    }

    public RevokeIds[] getRevokeIds() {
        return this.revokeIds;
    }

    public void setFirstChatCount(int i) {
        this.firstChatCount = i;
    }

    public void setInteractiveCount(int i) {
        this.interactiveCount = i;
    }

    public void setMessage(Message[] messageArr) {
        this.message = messageArr;
    }

    public void setNoticeCount(int i) {
        this.noticeCount = i;
    }

    public void setRejectIds(RevokeIds[] revokeIdsArr) {
        this.rejectIds = revokeIdsArr;
    }

    public void setRevokeIds(RevokeIds[] revokeIdsArr) {
        this.revokeIds = revokeIdsArr;
    }

    public String toString() {
        return "MessageModel{message=" + Arrays.toString(this.message) + ", rejectIds=" + Arrays.toString(this.rejectIds) + ", revokeIds=" + Arrays.toString(this.revokeIds) + ", firstChatCount=" + this.firstChatCount + ", interactiveCount=" + this.interactiveCount + ", noticeCount=" + this.noticeCount + '}';
    }
}
