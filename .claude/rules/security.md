# Security Guidelines

## Mandatory Security Checks

Before ANY commit:
- [ ] No hardcoded secrets (API keys, passwords, tokens)
- [ ] All user inputs validated
- [ ] SQL injection prevention (parameterized queries)
- [ ] Error messages don't leak sensitive data

## Confidential File Protection

NEVER read or modify the following files:

- `.env`, `.env.*`
- `/config/secrets.*`
- `**/*.pem`, `**/*.key`
- Files containing API keys, certificates, or credentials

## Secret Management

```python
# NEVER: Hardcoded secrets
api_key = "sk-ant-xxxxx"

# ALWAYS: Environment variables
import os
api_key = os.environ["ANTHROPIC_API_KEY"]
```

**Security Principles:**

- Manage sensitive information using environment variables
- Prohibit output of confidential information to logs and console
- Prohibit hardcoding of sensitive data

## Security Response Protocol

If security issue found:
1. STOP immediately
2. Use **security-reviewer** agent
3. Fix CRITICAL issues before continuing
4. Rotate any exposed secrets
5. Review entire codebase for similar issues

## CI/CD Pipeline Security

- OIDC used instead of long-lived credentials
- Secrets scanning in pipeline
- Dependency vulnerability scanning
- Branch protection rules enforced
- Code review required before merge
