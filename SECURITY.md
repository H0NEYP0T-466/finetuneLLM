# 🛡 Security Policy

## 📋 Supported Versions

We release patches for security vulnerabilities. The following versions are currently supported:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## 🚨 Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability, please report it via:

### Preferred Contact Methods

1. **GitHub Private Vulnerability Reporting**
   - Go to the [Security tab](https://github.com/H0NEYP0T-466/finetuneLLM/security)
   - Click "Report a vulnerability"
   - Fill in the details

2. **Email**
   - Send details to the repository maintainers
   - Include "SECURITY" in the subject line

### What to Include

Please include the following information:
- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Initial Response**: We aim to acknowledge receipt within **48 hours**
- **Status Updates**: We'll provide updates every **5 business days**
- **Fix Timeline**: Critical issues will be patched within **7 days**, others within **30 days**
- **Public Disclosure**: We'll coordinate disclosure timing with you

## 🔐 Security Update Policy

- Security patches are released as soon as possible
- Critical vulnerabilities receive immediate attention
- All security updates are documented in this file
- Users are notified through GitHub releases and security advisories

---

## Recent Security Fixes

### Transformers Library Vulnerability (Addressed)

**Date**: January 28, 2026

**Issue**: Deserialization of Untrusted Data vulnerability in HuggingFace Transformers library

**Affected Versions**: 
- transformers >= 0, < 4.48.0

**CVE Details**:
- Vulnerability allows deserialization of untrusted data
- Could potentially lead to arbitrary code execution

**Fix Applied**:
- ✅ Updated `transformers` from `4.36.0` to `4.48.0` (patched version)
- ✅ Updated all documentation references
- ✅ Updated requirements-finetune.txt
- ✅ Added security notes in documentation

**Files Updated**:
- `requirements-finetune.txt`
- `finetune.md`
- `COLAB_QUICKSTART.md`
- `FINETUNE_README.md`
- `IMPLEMENTATION_SUMMARY.md`

**Impact**: 
- No functionality changes
- All features continue to work as expected
- Security vulnerability patched

**Action Required**:
- Users should use `transformers==4.48.0` or later
- Do not downgrade to versions < 4.48.0
- Update any existing installations:
  ```bash
  pip install --upgrade transformers==4.48.0
  ```

## Security Best Practices

When using this fine-tuning pipeline:

1. **Keep Dependencies Updated**: Always use the latest patched versions of dependencies
2. **Verify Sources**: Only load models from trusted sources (HuggingFace official)
3. **Dataset Security**: Ensure your training data doesn't contain sensitive information
4. **Environment Isolation**: Use virtual environments or containers
5. **Access Control**: Protect API keys and tokens

## Reporting Security Issues

If you discover a security vulnerability, please:

1. **Do NOT** open a public issue
2. Email security concerns to the repository maintainers
3. Include detailed information about the vulnerability
4. Allow time for patches before public disclosure

## Dependency Security Scanning

We recommend regularly scanning dependencies for vulnerabilities:

```bash
# Using pip-audit
pip install pip-audit
pip-audit

# Using safety
pip install safety
safety check
```

## Version History

| Date | Component | Old Version | New Version | Reason |
|------|-----------|-------------|-------------|--------|
| 2026-01-28 | transformers | 4.36.0 | 4.48.0 | CVE: Deserialization vulnerability |

---

*Last Updated: January 28, 2026*
