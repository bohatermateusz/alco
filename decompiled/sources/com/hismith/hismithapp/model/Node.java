package com.hismith.hismithapp.model;

/* loaded from: classes3.dex */
public class Node {
    private boolean hasChild;
    private Integer id;
    private Integer level;
    private String name;
    private String pName;
    private Integer pid;

    public Node(Integer num, String str, Integer num2, String str2, Integer num3) {
        this.id = num;
        this.name = str;
        this.pid = num2;
        this.pName = str2;
        this.level = num3;
    }

    public Integer getId() {
        return this.id;
    }

    public Integer getLevel() {
        return this.level;
    }

    public String getName() {
        return this.name;
    }

    public Integer getPid() {
        return this.pid;
    }

    public String getpName() {
        return this.pName;
    }

    public boolean isHasChild() {
        return this.hasChild;
    }

    public void setHasChild(boolean z) {
        this.hasChild = z;
    }

    public void setId(Integer num) {
        this.id = num;
    }

    public void setLevel(Integer num) {
        this.level = num;
    }

    public void setName(String str) {
        this.name = str;
    }

    public void setPid(Integer num) {
        this.pid = num;
    }

    public void setpName(String str) {
        this.pName = str;
    }

    public String toString() {
        return "Node{id=" + this.id + ", name='" + this.name + "', pid=" + this.pid + ", level=" + this.level + ", hasChild=" + this.hasChild + '}';
    }
}
