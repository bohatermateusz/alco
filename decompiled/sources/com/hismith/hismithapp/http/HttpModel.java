package com.hismith.hismithapp.http;

/* loaded from: classes3.dex */
public class HttpModel {
    private Error[] errors;
    private int success;

    public class Error {
        private String errorCode;
        private String errorMessage;

        public Error() {
        }

        public String getErrorCode() {
            return this.errorCode;
        }

        public String getErrorMessage() {
            return this.errorMessage;
        }

        public void setErrorCode(String str) {
            this.errorCode = str;
        }

        public void setErrorMessage(String str) {
            this.errorMessage = str;
        }

        public String toString() {
            return "Error{errorCode='" + this.errorCode + "', errorMessage='" + this.errorMessage + "'}";
        }
    }

    public Error[] getErrors() {
        return this.errors;
    }

    public int getSuccess() {
        return this.success;
    }

    public void setErrors(Error[] errorArr) {
        this.errors = errorArr;
    }

    public void setSuccess(int i) {
        this.success = i;
    }
}
